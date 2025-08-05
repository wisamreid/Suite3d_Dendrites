"""
Profiler Module for Suite3D
============================

This module provides a lightweight, extensible instrumentation framework for timing,
memory, GPU usage, and shape inspection of Python functions in the Suite3D pipeline.
It is designed to be entirely optional during development, with zero dependencies on
external profiling tools (though it can leverage CuPy and psutil for enhanced metrics).

Core Functionalities:
---------------------
1) **Global Enable/Disable**
   - `enable_profiler()` and `disable_profiler()` toggle profiling on or off at runtime.
   - Controlled by the environment variable `SUITE3D_PROFILING` (default: "1" == enabled).

2) **Dynamic Log File Configuration**
   - `set_profiler_log_file(path: str)` closes the current log handle and opens a new one.
   - Logs are written as newline-delimited JSON to `LOG_FILE` (default file: `suite3d_profile_log.txt`).

3) **Decorator: `@profile(...)`**
   Wrap any function to capture:
     - **Wall-clock time** (`elapsed_s`)
     - **CPU memory delta** (`mem_delta_mb`) via `tracemalloc`
     - **GPU memory delta** and **peak** (`gpu_delta_mb`, `gpu_peak_mb`) via CuPy pool sampling
     - **Input and output shapes** or values via `_extract_shapes`

   **Parameters:**
   ```python
   @profile(
       timeit: bool = True,       # measure wall-clock time
       mem: bool    = True,       # measure Python-heap memory delta
       gpu: bool    = True,       # measure GPU pool usage
       shapes: bool = True,       # record input/output shapes or scalar values
       gpu_sample_interval: float = 0.05
   )
   ```

4) **Intermediate Logging**
   - `log_intermediate(**kwargs)` manually logs shapes/values of any local variables.
   - Prints to console and writes a JSON entry with key `'type': 'intermediate'`.

5) **Shape Extraction**
   - Handles NumPy/CuPy arrays (`.shape`, `.dtype`), generic sequences (`len()`), and basic
     scalars (`int`, `float`, `bool`, `str`).

Usage Example:
--------------
```python
# Pipeline script or demo
from suite3d.dev_utils.profiler import (
    set_profiler_log_file,
    enable_profiler,
    disable_profiler,
    profile,
    log_intermediate
)

# Optionally specify a custom log file anywhere in your script:
set_profiler_log_file("/absolute/path/to/my_profile_log.txt")

# Turn on profiling
enable_profiler()

@profile(timeit=True, mem=True, gpu=True, shapes=True)
def compute_heavy(job):
    # Inspect an intermediate array slice:
    intermediate = job.load_data()  # e.g. CuPy array
    log_intermediate(data_slice=intermediate[:10, :10])

    # Do work
    result = job.process(intermediate)
    return result

# Call your instrumented function:
output = compute_heavy(job)

# Turn off profiling
disable_profiler()
```

Log Entries:
------------
- **Profile entries** have keys:
  `'type': 'profile'`, `'function'`, `'elapsed_s'`, `'mem_delta_mb'`,
  `'gpu_delta_mb'`, `'gpu_peak_mb'`, `'input_shapes'`, `'output_shape'`, `'success'`, etc.
- **Intermediate entries** have `'type': 'intermediate'`, `'vars'` mapping var names to shapes/values.

Development Notes:
------------------
- Designed for minimal intrusion: simply decorate or call `log_intermediate` where needed.
- The module can be extended to include line-by-line stats, I/O profiling, or other metrics.
- By default, the log file is overwritten at import; use `set_profiler_log_file` to redirect mid-execution.

"""

import time
import threading
import functools
import os
import inspect
import json
import tracemalloc
import shutil

# Optional CPU memory via psutil
try:
    import psutil
    _process = psutil.Process()
except ImportError:
    _process = None

# Optional GPU memory via CuPy pool
try:
    import cupy as cp
    _gpu_pool_available = True
    _gpu_pool = cp.get_default_memory_pool()
except ImportError:
    _gpu_pool_available = False

# Optional NVML for true VRAM & utilization sampling
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
    nvmlInit()
    _nvml_handle = nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
except ImportError:
    _nvml_available = False

# Global profiling control and log path
PROFILE_ENABLED = os.getenv("SUITE3D_PROFILING", "1") == "1"
LOG_FILE = os.getenv("SUITE3D_PROFILE_LOG", "suite3d_profile_log.txt")

# Start tracemalloc for memory tracking and clear log
tracemalloc.start()
_log_fh = open(LOG_FILE, 'w', encoding='utf-8')


def parse_profiler_log(log_path, function_name=None, return_all=False):
    """
    Parse a profiler log file and return either:
    - The most recent matching profile record (default)
    - All matching profile records if `return_all=True`
    - All profile records (unfiltered) if `function_name=None`

    Args:
        log_path (str): Path to the profiler log (txt file with JSON lines).
        function_name (str, optional): Fully qualified name to filter for. If None, return all.
        return_all (bool): If True, return a list of all matching records.

    Returns:
        dict or list of dict: The most recent (or all) matching profile entries.
    """

    records = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") == "profile":
                    if function_name is None or entry.get("function") == function_name:
                        records.append(entry)
            except json.JSONDecodeError:
                continue  # skip corrupted lines

    if not records:
        raise ValueError(f"No matching profile records found in: {log_path}")

    if return_all:
        return records
    return records[-1]  # return most recent matching record


def extract_init_profiler_metrics(directory, profiler_log_name="profiler_log_init.txt", function=None):
    """
    Extract profiling metrics from a Suite3D profiler log.

    Looks for the most recent 'profile' entry (optionally filtered by function name)
    and extracts runtime, memory, and GPU usage metrics. This is useful for sweep
    summaries and benchmarking.

    Args:
        directory (str): Directory containing the profiler log file.
        profiler_log_name (str): Name of the profiler log file (default: 'profiler_log_init.txt').
        function (str, optional): Function name to filter by. If None, uses most recent profile entry.

    Returns:
        dict: Dictionary containing profiler metrics (e.g. runtime, memory delta, GPU peak).
    """
    log_path = os.path.join(directory, profiler_log_name)
    record = parse_profiler_log(log_path, function_name=function)

    return {
        "init_elapsed_s": record.get("elapsed_s"),
        "init_mem_delta_mb": record.get("mem_delta_mb"),
        "init_rss_delta_mb": record.get("rss_delta_mb"),
        "init_gpu_peak_mb": record.get("gpu_pool_peak_mb") or record.get("nvml_gpu_peak_mb"),
        "init_gpu_util_pct": record.get("nvml_gpu_peak_util_pct"),
    }


def set_profiler_log_file(path):
    """
    Dynamically change the profiler log file at runtime.
    Closes current file handle and opens a new one.
    """
    global LOG_FILE, _log_fh
    try:
        _log_fh.close()
    except Exception:
        pass
    LOG_FILE = path
    _log_fh = open(LOG_FILE, 'w', encoding='utf-8')


def enable_profiler():
    """Enable profiling globally."""
    global PROFILE_ENABLED
    PROFILE_ENABLED = True


def disable_profiler():
    """Disable profiling globally."""
    global PROFILE_ENABLED
    PROFILE_ENABLED = False


def _extract_shapes(obj):
    """
    Return descriptive info for obj:
      - .shape and .dtype for arrays
      - len() for sequences
      - value for scalars
    Convert shapes to lists and dtypes to strings for JSON.
    """
    try:
        if hasattr(obj, 'shape'):
            info = {'shape': list(obj.shape)}
            dtype = getattr(obj, 'dtype', None)
            if dtype is not None:
                info['dtype'] = str(dtype)
            return info
        if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes, dict)):
            return {'len': len(obj)}
        if isinstance(obj, (int, float, bool, str)):
            return {'value': obj}
    except Exception:
        pass
    return None


def log_intermediate(**kwargs):
    """
    Log intermediate vars shapes/values during execution.
    """
    if not PROFILE_ENABLED:
        return
    entry = {'type': 'intermediate', 'timestamp': time.time(), 'vars': {k: _extract_shapes(v) for k, v in kwargs.items()}}
    _log_fh.write(json.dumps(entry) + '\n')
    _log_fh.flush()
    parts = [f"{k}: {v}" for k, v in entry['vars'].items()]
    # get_terminal_size returns a tuple (columns, lines)
    width = shutil.get_terminal_size(fallback=(120, 24)).columns
    print("\n")
    print("─" * width)  # print a single full-width stripe
    print(f"[INTERMEDIATE] {'; '.join(parts)}")
    print("─" * width)  # print a single full-width stripe
    print("\n")


def profile(timeit=True, mem=True, gpu=True, shapes=True, gpu_sample_interval=0.05):
    """
    Decorator to measure execution metrics and log to file + console.
    Tracks:
      - wall-clock time
      - Python-level memory (tracemalloc)
      - per-process RSS (psutil)
      - CuPy pool allocations
      - NVML VRAM & GPU utilization
      - input/output shapes
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PROFILE_ENABLED:
                return func(*args, **kwargs)

            # --- Pre-call metrics ---
            t0 = time.perf_counter() if timeit else None
            mem0 = tracemalloc.take_snapshot() if mem else None
            rss0 = _process.memory_info().rss if _process else None

            # CuPy GPU pool sampler setup
            gpu0 = peak_gpu = None
            gpu_sampler = None
            if gpu and _gpu_pool_available:
                stop_evt = threading.Event()
                def _sample_pool():
                    nonlocal peak_gpu, gpu0
                    try:
                        gpu0 = _gpu_pool.used_bytes()
                        peak_gpu = gpu0
                    except Exception:
                        gpu0 = peak_gpu = 0
                    while not stop_evt.is_set():
                        try:
                            used = _gpu_pool.used_bytes()
                            if used > peak_gpu:
                                peak_gpu = used
                        except Exception:
                            pass
                        time.sleep(gpu_sample_interval)
                gpu_sampler = threading.Thread(target=_sample_pool, daemon=True)
                gpu_sampler.start()

            # NVML sampler setup
            nvml_sampler = None
            nvml_peak = {'mem': 0, 'util': 0}
            if gpu and _nvml_available:
                nvml_stop = threading.Event()
                def _sample_nvml():
                    nonlocal nvml_peak
                    info = nvmlDeviceGetMemoryInfo(_nvml_handle)
                    util = nvmlDeviceGetUtilizationRates(_nvml_handle)
                    nvml_peak['mem'] = info.used
                    nvml_peak['util'] = util.gpu
                    while not nvml_stop.is_set():
                        info = nvmlDeviceGetMemoryInfo(_nvml_handle)
                        util = nvmlDeviceGetUtilizationRates(_nvml_handle)
                        nvml_peak['mem'] = max(nvml_peak['mem'], info.used)
                        nvml_peak['util'] = max(nvml_peak['util'], util.gpu)
                        time.sleep(gpu_sample_interval)
                nvml_sampler = threading.Thread(target=_sample_nvml, daemon=True)
                nvml_sampler.start()

            # input shapes
            in_shapes = None
            if shapes:
                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
                in_shapes = {k: _extract_shapes(v) for k, v in bound.arguments.items()}

            # --- Function call ---
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception:
                success = False
                result = None
                raise
            finally:
                # Stop samplers
                if gpu and _gpu_pool_available and gpu_sampler:
                    stop_evt.set()
                    gpu_sampler.join()
                if gpu and _nvml_available and nvml_sampler:
                    nvml_stop.set()
                    nvml_sampler.join()

                # --- Post-call metrics ---
                t1 = time.perf_counter() if timeit else None
                mem1 = tracemalloc.take_snapshot() if mem else None
                rss1 = _process.memory_info().rss if _process else None
                gpu1 = None
                if gpu and _gpu_pool_available:
                    try:
                        gpu1 = _gpu_pool.used_bytes()
                    except Exception:
                        gpu1 = None

                # Build record
                record = {'type':'profile','function':func.__qualname__,'timestamp':time.time(),'success':success}
                console = [f"Function: {func.__qualname__}"]

                if timeit:
                    elapsed = t1 - t0
                    record['elapsed_s'] = elapsed
                    console.append(f"Time: {elapsed:.3f} s")
                if mem:
                    stats = mem1.compare_to(mem0, 'lineno')
                    mem_diff = sum(stat.size_diff for stat in stats)
                    record['mem_delta_mb'] = mem_diff/(1024**2)
                    console.append(f"Python Δ: {record['mem_delta_mb']:.3f} MiB")
                if _process:
                    rss_delta = (rss1 - rss0)/(1024**2)
                    record['rss_delta_mb'] = rss_delta
                    console.append(f"RSS Δ: {rss_delta:.3f} MiB")
                if gpu and _gpu_pool_available:
                    delta = (gpu1 - gpu0) if (gpu1 is not None and gpu0 is not None) else 0
                    record['gpu_pool_delta_mb'] = delta/(1024**2)
                    record['gpu_pool_peak_mb']  = (peak_gpu - gpu0)/(1024**2) if gpu0 is not None else peak_gpu/(1024**2)
                    console.append(f"CuPy Δ: {record['gpu_pool_delta_mb']:.3f} MiB")
                    console.append(f"CuPy Peak: {record['gpu_pool_peak_mb']:.3f} MiB")
                if gpu and _nvml_available:
                    info_end = nvmlDeviceGetMemoryInfo(_nvml_handle)
                    nvml_delta = info_end.used - nvml_peak['mem']
                    record['nvml_gpu_delta_mb']      = nvml_delta/(1024**2)
                    record['nvml_gpu_peak_mb']       = nvml_peak['mem']/(1024**2)
                    record['nvml_gpu_peak_util_pct'] = nvml_peak['util']
                    console.append(f"NVML Δ: {record['nvml_gpu_delta_mb']:.3f} MiB")
                    console.append(f"NVML Peak: {record['nvml_gpu_peak_mb']:.3f} MiB")
                    console.append(f"NVML Util: {record['nvml_gpu_peak_util_pct']:.1f}%")
                if shapes:
                    out_shape = _extract_shapes(result)
                    record['input_shapes'] = in_shapes
                    record['output_shape'] = out_shape
                    console.append(f"Inputs: {in_shapes}")
                    console.append(f"Output: {out_shape}")

                # Write JSON log
                _log_fh.write(json.dumps(record)+'\n')
                _log_fh.flush()

                # Print summary
                width = shutil.get_terminal_size(fallback=(140,24)).columns
                print("\n" + "─"*width)
                print(f"[PROFILE] {' | '.join(console)}")
                print("─"*width + "\n")
            return result
        return wrapper
    return decorator
