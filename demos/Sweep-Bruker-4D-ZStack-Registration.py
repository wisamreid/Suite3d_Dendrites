#!/usr/bin/env python3
"""
Bruker-4D-ZStack-Registration.py

A self-contained demonstration of Suite3D’s registration pipeline,
complete with nonrigid correction, vector-field visualization, and automated
benchmarking.  Designed to run on a small “standard-2p” dataset of two TIFFs
and illustrate the full workflow—from raw movie → registered output → plots,
GIFs, BigTIFF exports, and quantitative comparisons—without modification.

This script runs Suite3D’s 3D + per-plane nonrigid registration pipeline
on the provided dataset (Z:\Wisam\Z_Python\Suite3D Data\standard-2p)
complete with:

  • init_pass → builds a 3D reference and computes plane offsets
  • register   → rigid 3D alignment + 2D nonrigid brick-wise warps
  • per-plane fused output chunks (NumPy + BigTIFF)
  • Diagnostic plots (global rigid vs. per-plane nonrigid, block-wise timecourses,
    vector-field overlays, plane-alignment offsets)
  • (optional) Animated GIFs of the nonrigid vector field over time
  • Benchmarks of each step against a saved “baseline” run

USAGE
-----
    python Demo-Standard-Data-Register-Only.py

MODES OF OPERATION
------------------
This script operates in **three** main modes, controlled by the `create` and
`overwrite` flags (and optionally `reset_benchmark_baselines`):

  1. **Baseline generation**
     - `create=True, overwrite=True`
     - *What happens:*
       • Creates a fresh `<output_root>/<job_name>/` folder.
       • Runs `init_pass()` → saves `summary.npy`.
       • Runs `register()` → saves chunked `.npy` (`registered_fused_data/`).
       • Computes shifts, vector fields, GIFs, plots → saves under
         `registration_shifts/`.
       • Writes per-plane BigTIFF stacks under `registered_tiffs/`.
       • **Bootstraps** all benchmarks (stored under `benchmarks/<step>/baseline/`).

  2. **Plot-only rerun**
     - `create=False, overwrite=False`
     - *What happens:*
       • Detects existing `summary.npy` and fused-data chunks → skips both
         `init_pass()` and `register()`.
       • Reloads saved arrays and regenerates **all** plots, GIFs, and
         **comparisons** against the saved baseline.

  3. **Full rerun + comparative benchmarking**
     - `create=True, overwrite=True`  *(with `reset_benchmark_baselines=False`)*
     - *What happens:*
       • Recomputes `init_pass()` and `register()` end-to-end (overwriting
         previous outputs).
       • Reloads existing baseline under `benchmarks/*/baseline/` → runs
         side-by-side comparisons, saving the results in new timestamped
         subdirectories under `benchmarks/<step>/`.

OPTIONAL FLAG
-------------
  reset_benchmark_baselines (bool)
    If True, *deletes* all existing `benchmarks/*/baseline/` directories
    **before** running any benchmarks, so that the next save always
    bootstraps a fresh baseline.

EXPECTED INPUT (bruker_data_dir)
------------------------------
Z:\Wisam\Z_Python\Suite3D Data\TSeries-03062025-8sxAA091024M1-009\
├── TSeries-03062025-8sxAA091024M1-009_Cycle00001_Ch1_000001.ome.tif
└── TSeries-03062025-8sxAA091024M1-009_Cycle00001_Ch1_000002.ome.tif
└── ...
├── TSeries-03062025-8sxAA091024M1-009_Cycle00001_Ch2_000001.ome.tif
└── TSeries-03062025-8sxAA091024M1-009_Cycle00001_Ch2_000002.ome.tif
└── ...
├── TSeries-03062025-8sxAA091024M1-009_Cycle00002_Ch1_000001.ome.tif
└── TSeries-03062025-8sxAA091024M1-009_Cycle00002_Ch1_000002.ome.tif
└── ...
├── TSeries-03062025-8sxAA091024M1-009_Cycle00002_Ch2_000001.ome.tif
└── TSeries-03062025-8sxAA091024M1-009_Cycle00002_Ch2_000002.ome.tif
└── ...

EXPECTED OUTPUT (`<output_root>/TEST/`)
---------------------------------------
1) **First run** (`create=True, overwrite=True`)

    SUITE3D_TEST_OUTPUT/TEST/
    ├── summary/
    │   └── summary.npy
    ├── init_summary/
    │   ├── ref_img_3d.npy
    │   └── plane_shifts.npy
    ├── registered_fused_data/
    │   ├── fused_reg_data0000.npy
    │   └── fused_reg_data0001.npy
    ├── registration_shifts/
    │   ├── mean_fused_volume.npy
    │   ├── offset_dict.npz
    │   ├── ymaxs_nr_blocks.npy
    │   ├── xmaxs_nr_blocks.npy
    │   ├── vector_field_over_time_plane0.gif
    │   ├── vector_field_over_time_plane1.gif
    │   └── … one GIF per plane
    ├── registered_tiffs/
    │   ├── plane0/
    │   │   └── registered_plane0_1-100.tif
    │   └── plane1/…
    └── benchmarks/
        ├── initialization/
        │   └── baseline/    ← baseline JSON + .npz
        └── registration/
            └── baseline/    ← baseline JSON + .npz
        └── (Optional) corr_map/
            └── baseline/    ← baseline JSON + .npz

2) **Plot-only rerun** (`create=False, overwrite=False`)

    *No changes to any data files under* `TEST/` (init & reg skipped)

3) **Full rerun + comparative benchmarking** (`create=True, overwrite=True`)

    *All* `summary/`, `registered_fused_data/`, `registration_shifts/`,
    and `registered_tiffs/` are **re-generated**.
    `benchmarks/` retains each step’s `baseline/` subfolder and adds a
    fresh `<timestamp>/` comparison folder alongside it.
    `benchmarks/` now contains:
    ├── initialization/
    │   ├── baseline/          ← original baseline
    │   └── <timestamp>/       ← new comparison outputs
    └── registration/
        ├── baseline/          ← original baseline
        └── <timestamp>/       ← new comparison outputs

KEY POINTS TO REMEMBER
----------------------

  • **Extended-length Windows paths** (“\\\\?\\…”) are used under the hood
    to avoid MAX_PATH issues on Windows. You should never need to worry
    about it unless you change the base directory.

  • **Tile grid** for nonrigid warps is pulled **directly** from
    `job.load_summary()["reference_params"]["yblock"]` and
    `["xblock"]`—so you always reshape using the exact same
    (nYb × nXb) tiling as init_pass.

  • **Block grid** (`nYb×nXb`) is loaded directly from
    `summary["reference_params"]["yblock"]` & `["xblock"]`—don’t hard-code it.

  • **Flatten ↔ reshape**: raw per-chunk shifts come in shape
    `(n_chunks, T, P, nBlocks)`; collapse & reshape to `(T, P, nYb, nXb)`.

  • **Rigid vs. nonrigid**: we first apply a 3D shift to every volume,
    then per-plane 2D “block-wise” nonrigid shifts. Plots and exports
    separate these two passes for clarity.

  • **Benchmarking**: on first run, each step writes a “baseline” JSON
    and NumPy dump. On subsequent runs (without baseline present), it compares
    the new outputs to the baseline to ensure byte-for-byte or
    numeric consistency.

  • **Plotting & GIFs**: we always overlay on the _mean fused volume_
    (not a single raw frame), and we auto-scale arrows so that the
    largest shift spans ~20 px by default. You can tweak
    `scale_factor` or `arrow_length` in the “Mean 2D vector-field
    overlay” block to suit your preferences.

  • **Flags:**
    - `create` controls directory creation.
    - `overwrite` controls whether to *re-compute* expensive steps.
    - `reset_benchmark_baselines` (optional) automates deletion of all `baseline/` folders.

Now—scroll down past this docstring to the imports and main-body.
"""

# TODO: Add benchmark debugging by flagging an expanded exception in their try-exception blocks

# TODO: toggle off benchmarking when running plots only

# TODO: I modified suite3d.iter_step.register_dataset_gpu_3d in order to implement 2D nonrigid
#   registration. Currently this makes the pipeline insensitive to the params['nonrigid'] flag.
#   This should be handled by job.register() but should I build it into my version of register_dataset_gpu_3d?

# TODO: Improve the relevance of the background images for all vector field outputs
#   e.g., the vector field over time gifs should be overlayed on the movie.

# TODO: Re-evaluate which benchmarks we want to look at for initialization and registration
#   There are several well‐established, “blind” (i.e. reference‐free or reference‐light) metrics
#       Each frame vs. a reference image (e.g. the mean or max projection)
#       Each frame vs. its immediately neighboring frame (to see how much “jitter” remains after registration)
#       1. Normalized Cross‐Correlation (NCC):
#           Measures how well each frame matches the 3D reference image or the temporal mean image.
#           Values ∈ [–1, 1], where 1 is perfect match.
#           Scale‐ and offset‐invariant, easy to compute via FFT or direct sliding‐window.
#           Compute: corr = np.corrcoef(ref_img.ravel(), frame.ravel())[0,1]
#           record the per‐frame NCC array, or simply its mean and std across frames
#       2. Structural Similarity Index (SSIM):
#           Measures perceptual similarity that accounts for luminance, contrast, and structure
#           More sensitive to local distortions than raw correlation or MSE
#           Compute: via skimage.metrics.structural_similarity
#           score, _ = structural_similarity(ref_img, frame, full=True)
#           record the median SSIM across time as a single scalar
#       3. Mean-Squared Error (MSE) or Root-Mean-Squared Error (RMSE)
#           Measures pixel‐wise error between frame and reference
#           Extremely simple; a reduction in MSE after registration is direct evidence of improved alignment
#           report the mean MSE (or RMSE) before vs. after registration.
#       4. Frame-to-Frame Residual Motion
#           Measures after registration, how much each frame still differs from its temporal neighbors
#           Directly captures “jitter” that registration failed to remove
#           Compute: diffs = np.sqrt(((frame[t] - frame[t-1])**2).mean())
#           report mean or median of these residuals across the entire movie.
#       5. Mutual Information (MI)
#           Measures statistical dependence between intensities of frame and reference—robust to non‐linear intensity mappings.
#           Often used in multi‐modal registration (e.g. MRI vs. CT), but can pick up subtle misalignments
#           Compute: skimage.metrics.normalized_mutual_information or histogram‐based estimation
#   Other possible per-run scalar or small-array quality metrics:
#       Fraction of pixels above some sharpness threshold

import os
import glob
import time
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tifffile as tiff
from suite3d.job import Job
import suite3d.io as io_module  # for get_tif_paths, get_vol_rate
from suite3d.utils import get_repo_status, benchmark, save_benchmark_results, find_git_root
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import shutil
from datetime import datetime

# from tifffile import TiffWriter, imwrite
# from tqdm import trange
# from PIL import Image

# For currently unused GIF generation helper functions
from skimage.transform import resize
import re

from dask import diagnostics

from suite3d.utils import choose_gpu_reference_batch_size, bytes_gib

# NOTE: Code for gpu_reference_batch_size selection
# # Example: suppose you’ll process (after pad/crop) nz=229, ny=470, nx=490
# nz, ny, nx = 229, 470, 490
#
# # Get available VRAM (MiB→bytes). If you have NVML, you can read free VRAM.
# # Otherwise, set manually (e.g., 7.0 GiB headroom on an 8 GiB card).
# available_vram_bytes = int(7.0 * (1024**3))  # 7 GiB budget
#
# desired_nt = job.params["n_init_files"]  # or any target upper bound
# safe_nt, est_bytes = choose_gpu_reference_batch_size(
#     nz, ny, nx,
#     available_vram_bytes=available_vram_bytes,
#     max_nt=desired_nt,
#     safety_mult=3.0,         # raise to 3.5–4.0 if you still hit OOM
# )
#
# print(f"[ref3d] chosen gpu_reference_batch_size={safe_nt} "
#       f"(est peak ~ {bytes_gib(est_bytes):.2f} GiB for {nz}×{ny}×{nx})")
#
# job.params["gpu_reference_batch_size"] = max(1, safe_nt)


"""
    Top-Level Parameters
"""

# ─── Modular Execution Flags ─────────────────────────────────────────────────────────────────
run_init = True                     # Run init_pass() if True
run_register = True                 # Run job.register() if True

run_export_tiffs_by_planes = False  # Write TIFFs to disk
max_tiff_export_workers = None      # Set to None for max(1, cpu_count - 1)

run_export_zstack = True                        # set False to skip
zstack_projection_types = ["mean", "median"]    # Options ["mean", "median", "max"], set to None for default ["mean"]

""" Development """

# ─── Parameter Sweeps ────────────────────────────────────────────────────────────────────────
run_reference_sweep = True      # ← Set to True to activate reference param sweep
# reference_sweep_params = {
#                 # Example sweep params — update as needed
#                 "smooth_sigma_time": [0.5, 1.0, 1.5],
#                 "smooth_sigma_space": [0.5, 1.0],
#             }
# reference_sweep_params = {
#     "block_size": [
#         [64, 64],
#         [128, 128],  # baseline setting
#     ]
# }
reference_sweep_params = {
    # "n_init_files": [50, 200, 500]
    "n_init_files": [100, 200, 500]
}
# reference_sweep_params = {
#     "gpu_reference_batch_size": [1, 4, 8, 20, 50]
# }
run_registration_sweep = False   # ← Set to True to activate registration param sweep

# ——— Benchmark management (these flags apply to both ) ———
run_benchmarking = True                     # Run benchmarking comparisons (init, reg, corr_map)
benchmark_tag = "test-sweep-229p-500frames"    # Set to a descriptive string or use datetime if set to None
reset_benchmark_baselines = False           # If True, delete all benchmark baselines

# ——— Profiling management ———
profiling = True
if profiling:
    from suite3d.dev_utils.profiler import enable_profiler, disable_profiler, set_profiler_log_file
profiler_log_file = "profiler_log.txt"

# ——— Debug management ———
debug_demo = False                          # Control local prints
run_diagnostics = True                      # Run plotting and summary diagnostics
generate_nonrigid_vector_field_gifs = True
gif_fps = 30                                # how many frames per second in the GIFs

# ——— 3D Correlation Map ———
# compute 3D correlation map
# NOTE: Not tested
run_corr_map = False                # Optional correlation map step

""" File Paths """

demo_data_base = r"Z:\Wisam\Z_Python\Suite3D Data"
demo_data_dir = demo_data_base + r"\TSeries-03062025-8sxAA091024M1-009"
output_base_raw = demo_data_base + r"\SUITE3D_TEST_OUTPUT"
job_name = "N_INIT_FILES-TUNING-TSeries-03062025-8sxAA091024M1-009"
# For speed set to a small integer (e.g., 2), set to None to run all TIFFs
max_num_tiffs_to_use = None # TODO: Figure out what to do with this

""" Suite3D Parameters """

create = True
overwrite = True
verbosity = 3                       # Verbosity level. 0: critical only, 1: info, 2: debug. Defaults to 1
nonrigid = True                     # Run plane-wise nonrigid registration after rigid registration
init_file_sample_method = 'even'    # 'even' or 'random' sampling of init files

# ——— TIFF Metadata ———
# voxel size in z,y,x in microns
voxel_size_um = (0.718, 0.8077042, 0.8077042)
functional_color_channel = 0    # which color channel is the functional one
num_colors = 2                  # how many color channels were recorded by scanimage
tau = 0.2                       # Decay time of the Ca indicator in seconds. 1.3 for GCaMP6s. The deme is for GCamP8m
num_planes =  229               # Number of planes recorded by Bruker, including the flyback
planes = range(num_planes)      # Planes to analyze (flyback planes are typically plane 0)
max_reg_tif_size = 500          # maximum size for output registered TIFF writing (set to 500 later)
fs = 30.0                       # this is the actual framerate in PrairieView (Hz)

# ——— Memory Management ———
split_tif_size = 100    # NOTE: If you have large tiffs, split the large tiffs into files of size 100 after registration
# require exact frame‐counts so preregistration can’t mis-fire
# limit how many TIFFs get loaded at once (reduces host‐RAM spikes)
# [DEFAULT] = 2
# tif_batch_size = 1
# NOTE: extra parameter for convenience
volumes_per_batch = 10  # How many *volumes* you want per GPU batch
# Each volume consists of one TIFF per (plane × color), so:
#   num_planes = 2
#   num_colors = 2
# → 2 × 2 = 4 TIFFs per volume
# To get 10 volumes per batch, you need 4 × 10 = 40 TIFFs:
tif_batch_size = num_planes * num_colors * volumes_per_batch

# Reference‐image smoothing/blocking ———
# how big each block is when computing masks & FFTs
# smaller → less peak memory, but more boundary overhead
# NOTE: the initial pass and reference building is NOT sensitive
#  to this parameter
# [DEFAULT] = [128, 128]
block_size = [128, 128]
# block_size = [64, 64]
# block_size = [32, 32]
# how many planes to batch through the GPU when building the reference
# smaller → less VRAM, but more kernel launch overhead
# [DEFAULT] = 20
gpu_reference_batch_size = 8
# gpu_reference_batch_size = 100
# number of files to use for the initial pass
# ~500 frames is a good rule of thumb
# we will just use 200 here for speed
n_init_files = 100   # NOTE: This now means how many frames/full volumes
# TODO: Determine the right way to generalize this
init_n_frames = None
# require exact frame‐counts so preregistration can’t mis-fire
tif_preregistration_safe_mode = True
fix_shallow_plane_shift_estimates = False


# TODO: Figure out if computing the correlation map is appropriate in this script
#   What parameters should be used here if so?
# Corr_map parameters (optional)
compute_corr_map         = False        # set False to skip
conv_filt_type           = 'gaussian'
conv_filt_xy             = 1.0
conv_filt_z              = 1.0
npil_filt_type           = 'unif'
npil_filt_xy             = 5.0
npil_filt_z              = 1.5
sdnorm_exp               = 0.8
intensity_thresh         = 0.2
t_batch_size_corr        = 100
n_proc_corr              = 10
mproc_batchsize          = 1


""" Currently Unused Helper Functions """

def _normalize_array(array, dtype):
    """
    Normalize an array to the range supported by the given dtype.

    Parameters:
    - array: numpy.ndarray, the array to normalize.
    - dtype: numpy.dtype, the target data type.

    Returns:
    - numpy.ndarray: The normalized array.
    """
    if dtype is None:
        dtype = array.dtype

    if np.issubdtype(dtype, np.integer):
        # Normalize to the integer range [0, max]
        info = np.iinfo(dtype)
        array = ((array - array.min()) / (array.max() - array.min()) * info.max).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        # Normalize to the floating-point range [0.0, 1.0]
        array = ((array - array.min()) / (array.max() - array.min())).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype for normalization: {dtype}")

    return array


def _resize_frame(frame, scale_factor):
    """
    Resize a frame without altering its original values.

    Parameters:
    - frame: 2D or 3D numpy array, the frame to resize.
    - scale_factor: float, the scale factor to apply.

    Returns:
    - Resized frame
    """
    if scale_factor != 1.0:
        frame = resize(frame,
                      (int(frame.shape[0] * scale_factor), int(frame.shape[1] * scale_factor)),
                      order=1, preserve_range=True).astype(frame.dtype)
    return frame


def save_array_to_gif(array, output_gif_path, fps=10, scale_factor=1.0, loop=0, dtype=np.uint8, normalize=True):
    """
    Converts a 3D or 4D array into a 2D or color animated GIF.

    Parameters:
    ----------
    array : numpy.ndarray
        The array to convert. Shape can be:
        - (N, H, W) for grayscale GIFs.
        - (N, H, W, C) for color GIFs (e.g., RGB with C=3).
    output_gif_path : str
        Path to save the output GIF file.
    fps : int, optional
        Frames per second for the GIF. Determines playback speed.
    scale_factor : float, optional
        Scale factor to resize each frame (1.0 retains original size).
    loop : int, optional
        Number of times the GIF will loop:
        - loop=0 (default): Loop indefinitely.
        - loop=1: Play once (no looping).
        - loop=N: Loop N times, then stop.
    dtype : numpy.dtype, optional
        Data type for the output frames (e.g., np.uint8 for 8-bit GIFs).
    normalize : bool, optional
        Whether to normalize the array values to fit the specified dtype range.

    Returns:
    -------
    None
        Saves the GIF to the specified output path.
    """
    try:
        # Debug: Check the data type and range of the input array
        print(f"Array dtype: {array.dtype}, shape: {array.shape}")
        print(f"Array min: {array.min()}, max: {array.max()}")

        # Normalize if required
        if normalize:
            array = _normalize_array(array, dtype=dtype)

        # Process each frame
        frames = []
        for i in range(array.shape[0]):  # Iterate through the time dimension
            frame = array[i]
            frame = _resize_frame(frame, scale_factor)
            frames.append(frame.astype(dtype))  # Ensure the frame has the correct dtype

        # Save as a GIF
        imageio.mimsave(output_gif_path, frames, fps=fps, loop=loop)
        print(f"GIF saved to {output_gif_path}")

    except Exception as e:
        print(f"Error: {e}")


""" Helper Functions """


# ─── Safe Dask Chunking Before TIFF Export ──────────────────────────────
def safe_rechunk_for_tiff_export(mov_reg, verbose=True):
    """
    Ensures mov_reg is chunked as (1 plane, all time, full Y, full X).
    """
    if not hasattr(mov_reg, "chunks"):
        raise TypeError("Expected a Dask array with .chunks attribute")

    z_chunks, t_chunks, y_chunks, x_chunks = mov_reg.chunks

    if verbose:
        print("\n📦 Original Dask chunks:")
        print(f"  Z: {z_chunks}")
        print(f"  T: {t_chunks}")
        print(f"  Y: {y_chunks}")
        print(f"  X: {x_chunks}")

    n_planes, n_frames = mov_reg.shape[:2]

    try:
        mov_reg = mov_reg.rechunk((1, n_frames, -1, -1))
        if verbose:
            print("✅ Rechunked successfully for per-plane TIFF export:")
            print(f"  New Z chunks: {mov_reg.chunks[0]}")
            print(f"  New T chunks: {mov_reg.chunks[1]}")
        return mov_reg

    except MemoryError as e:
        print("❌ MemoryError during rechunking — will attempt lazy fallback export.")
        raise RuntimeError("Rechunking failed. Try reducing export size or batch size.") from e

    except Exception as e:
        print(f"❌ Unexpected error during rechunking: {e}")
        raise


def write_registered_planes_parallel(
    mov_reg,
    output_dir: str,
    max_tiff_pages: int = 500,
    max_workers: int = None
):
    """
    Write Suite3D’s registered movie to disk as per‐plane, chunked TIFFs, in parallel.

    Workflow for each imaging plane p (axis 0 of mov_reg):
      1. Create subdirectory `plane{p}` under `output_dir` (already in extended‐length form on Windows).
      2. Split the time axis (axis 1) into contiguous chunks of up to `max_tiff_pages` frames.
      3. For each chunk [start:end]:
         a. Compute the Dask slice mov_reg[p, start:end] → (chunk_size, H, W) NumPy array.
         b. Write that 3D NumPy array as a multi‐page BigTIFF named
            `registered_plane{p}_{start+1}-{end}.tif`.
      4. Run all planes in parallel using up to `max_workers` threads.

    On Windows, every `out_path` is already assumed to begin with “\\\\?\\” so that
    Win32 open( ) will accept long network or drive‐letter paths.

    Parameters
    ----------
    mov_reg : dask.array.Array or array‐like
        Registered movie with shape (n_planes, total_frames, height, width). Typically
        returned by `job.get_registered_movie(key="registered_fused_data", …)`.  If
        it’s a Dask array, each `.compute()` pulls only the requested slab into memory.

    output_dir : str
        Base directory under which subdirectories `plane0/`, `plane1/`, … are created.
        Must already be an “extended‐length” path on Windows (e.g., “\\\\?\\Z:\\…”).

    max_tiff_pages : int, default 500
        Maximum number of time frames (pages) per output TIFF.  If a plane has more than
        `max_tiff_pages` frames, it will be split into multiple TIFFs of up to that many pages.

    max_workers : int or None, default None
        Maximum number of threads to use for parallel writing.  If None, defaults to
        one thread per plane (i.e., n_planes threads).  Recommended to set to
        min(n_planes, CPU_count).

    Returns
    -------
    None
        Creates directories and writes TIFFs to disk.  Prints progress messages to stdout.
    """
    base_dir = Path(output_dir)
    # Create the base output directory (extended‐length form if on Windows)
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create base output directory {base_dir!r}") from e

    n_planes, total_frames, _, _ = mov_reg.shape
    workers = max_workers or min(n_planes, os.cpu_count() or 1)

    # Pre‐create each plane’s subdirectory once
    plane_dirs = []
    for p in range(n_planes):
        plane_folder = base_dir / f"plane{p}"
        try:
            plane_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Could not create plane directory {plane_folder!r}") from e
        plane_dirs.append(plane_folder)

    def _write_plane(p: int):
        """
        Worker to write all chunks for plane p. Subdirectory `plane{p}` already exists.
        """
        plane_folder = plane_dirs[p]

        for start in range(0, total_frames, max_tiff_pages):
            end = min(start + max_tiff_pages, total_frames)
            page_start = start + 1  # 1-based for filenames
            page_end = end

            fname = f"registered_plane{p}_{page_start}-{page_end}.tif"
            out_path = plane_folder / fname

            # 1) Compute the slab (frames → NumPy)
            slab = mov_reg[p, start:end]
            try:
                slab_np = slab.compute() if hasattr(slab, "compute") else np.asarray(slab)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compute slab for plane {p}, frames {start}-{end}"
                ) from e

            # 2) Write the chunk as a BigTIFF
            try:
                tiff.imwrite(
                    str(out_path),
                    slab_np,
                    photometric="minisblack",
                    metadata={'axes': 'TYX'},
                    bigtiff=True,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to write TIFF: {out_path}") from e

        return p  # Return plane index for logging

    # Launch one thread per plane (or up to `workers`)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_write_plane, p): p for p in range(n_planes)}
        for future in as_completed(futures):
            p_idx = futures[future]
            try:
                _ = future.result()
                print(f"✅ Plane {p_idx} complete")
            except Exception as exc:
                print(f"❌ Plane {p_idx} failed: {exc}")
                raise

    print("All planes written successfully.")


def frame_plane_chan_key(pth):
    """
    Sort key for Bruker single-page TIFFs so that any slice of the list
    truly interleaves (frame → plane → channel).
    """
    base = os.path.basename(pth)
    m = re.search(r"_Cycle(\d+)_Ch(\d+)_(\d+)\.ome\.tif$", base)
    if not m:
        return (0, 0, 0)
    # note: capture groups are (plane, channel, frame)
    plane = int(m.group(1))
    chan  = int(m.group(2))
    frame = int(m.group(3))
    return (frame, plane, chan)


def main():

    """
    Main entry point. Finds TIFFs in the demo folder, runs Suite3D’s init_pass()
    and register() only if necessary, then writes per‐plane, chunked TIFFs.
    """

    t_start = time.time()

    if max_tiff_export_workers is not None:
        num_tiff_export_workers = max_tiff_export_workers
    else:
        cpu_count = multiprocessing.cpu_count()  # total logical cores
        num_tiff_export_workers = max(1, cpu_count - 1)  # leave one core free

    # ─── Setup: ──────────────────────────────────────────────────────────
    # Base directories (mapped drive “Z:\…”). We will prefix with “\\\\?\\” below.
    # NOTE: See top of the script

    # Get the first N demo TIFFs (for speed); remove “[:N]” to run all 10
    if max_num_tiffs_to_use is not None:
        tifs = io_module.get_tif_paths(demo_data_dir)[:max_num_tiffs_to_use]
    else:
        tifs = io_module.get_tif_paths(demo_data_dir)
    for tif in tifs: print(tif)
    tifs.sort(key=frame_plane_chan_key) # TODO: Decide if this is needed

    # ──────────────────────────────────────────────────────────────────────────────
    # 1. Assemble Job Parameters
    # ──────────────────────────────────────────────────────────────────────────────

    # Suite3D parameters
    params = {
        "bruker": False, # NOTE: Convert single-page Bruker TIFFs into multipage tiffs before loading
        'bruker_lazy': True,    # NOTE: Use in‐memory Dask loader for Bruker data
        # volume rate
        # 'fs': io_module.get_vol_rate(tifs[0]), # TODO: Does this apply to Bruker data in some way?
        'fs': fs / num_planes,    # Compute "volume rate" (fs). NOTE: This doesn't actually matter
        # planes to analyze. 0 is typically the flyback, so we exclude it here
        'planes': planes,
        # number of planes recorded by scanimage, including the flyback
        'n_ch_tif': num_planes,
        # Decay time of the Ca indicator in seconds. 1.3 for GCaMP6s. This example is for GCamP8m
        'tau': tau,
        'lbm': False,
        'num_colors': num_colors,    # how many color channels were recorded by scanimage
        'functional_color_channel': functional_color_channel,  # which color channel is the functional one
        # voxel size in z,y,x in microns
        'voxel_size_um': voxel_size_um,
        # number of files to use for the initial pass
        # usually, ~500 frames is a good rule of thumb
        # we will just use 200 here for speed
        'n_init_files': n_init_files,
        # 3D GPU registration - fast!
        '3d_reg': True,
        'gpu_reg': True,
        # NOTE: 3D CPU is not supported yet
        'subtract_crosstalk': False,  # turn off some lbm-only features
        'fuse_strips': False,  # turn off some lbm-only features
        # ——— Reference‐image smoothing/blocking ———
        # how big each block is when computing masks & FFTs
        # smaller → less peak memory, but more boundary overhead
        'block_size': block_size,
        # how many planes to batch through the GPU when building the reference
        # smaller → less VRAM, but more kernel launch overhead
        'gpu_reference_batch_size': gpu_reference_batch_size,
        # require exact frame‐counts so preregistration can’t mis-fire
        'tif_preregistration_safe_mode': tif_preregistration_safe_mode,
        # limit how many TIFFs get loaded at once (reduces host‐RAM spikes)
        'tif_batch_size': tif_batch_size,
        'nonrigid': nonrigid,   # Run nonrigid registration?
        "init_n_frames": init_n_frames,
        "fix_shallow_plane_shift_estimates": fix_shallow_plane_shift_estimates,
        # (Optional) Corr_map parameters—only used if compute_corr_map=True
        # 'conv_filt_type': conv_filt_type,
        # 'conv_filt_xy': conv_filt_xy,
        # 'conv_filt_z': conv_filt_z,
        # 'npil_filt_type': npil_filt_type,
        # 'npil_filt_xy': npil_filt_xy,
        # 'npil_filt_z': npil_filt_z,
        # 'sdnorm_exp': sdnorm_exp,
        # 'intensity_thresh': intensity_thresh,
        # 't_batch_size': t_batch_size_corr,
        # 'n_proc_corr': n_proc_corr,
        # 'mproc_batchsize': mproc_batchsize,
    }

    # Compute “extended‐length” root on Windows by prefixing “\\\\?\\”
    if os.name == "nt":
        extended_root = "\\\\?\\" + output_base_raw
    else:
        extended_root = output_base_raw

    # ──────────────────────────────────────────────────────────────────────────────
    # 2. Create the Suite3D Job (create=False, overwrite=False)
    # ──────────────────────────────────────────────────────────────────────────────

    """
        Instantiate Job WITH or WITHOUT forcing recreation (fast‐path if results exist)
          root_dir: pick a base folder for all your runs (this can be anywhere you have write access)
          job_name: give your run a name (this will create: <output_base>\<job_name>)
          tifs: this is your list of .tif paths; params is the dict of parameters you set up
          create=True: tells it to mkdir the folder if it doesn’t exist
          overwrite=True: lets it delete an old run of the same name
          verbosity: sets logging detail level to the console
    """
    job = Job(
        extended_root,  # root_dir
        job_name,  # job name (positional!)
        tifs=tifs,  # list of TIFF paths
        params=params,  # params dict
        create=create,
        overwrite=overwrite,
        verbosity=verbosity
    )

    # ─── clamp the Z‐extent of the reference half‐window on the Job ────
    if "pc_size" in job.params:
        # unpack (z_half, y_half, x_half)
        z_pc, y_pc, x_pc = job.params["pc_size"]
        # limit z_pc to at most floor(n_planes/2)
        max_z = job.params["n_ch_tif"] // 2
        z_pc = min(z_pc, max_z)
        job.params["pc_size"] = np.array((z_pc, y_pc, x_pc), dtype=int)
        job.log(f"Clamped pc_size → (z={z_pc}, y={y_pc}, x={x_pc})", 2)

        # if you want to persist this change back to disk:
        job.save_params()

    # ─── Ensure the necessary subfolders exist when create=False ─────────────────
    # If this is the *first* time, create=False → none of these exist yet, so we must mkdir.
    # If this is a rerun, and they already exist, exist_ok=True means “do nothing.”
    os.makedirs(job.dirs["job_dir"], exist_ok=True)
    os.makedirs(job.dirs["summary"], exist_ok=True)
    os.makedirs(job.dirs["registered_fused_data"], exist_ok=True)
    os.makedirs(job.dirs["iters"], exist_ok=True)

    # ─── set up benchmarks ────────────────────────────────────────────────────────
    results_dir = Path(job.dirs["job_dir"]) / "benchmarks"
    results_dir.mkdir(exist_ok=True, parents=True)

    # find the git repo root automatically
    repo_path = find_git_root()  # assumes you’ve defined find_git_root()
    repo_status = get_repo_status(repo_path)

    # ─── optional: reset all baselines ───────────────────────────────────────────
    if reset_benchmark_baselines:
        for step in ("initialization", "registration", "corr_map"):
            bdir = results_dir / step / "baseline"
            if bdir.exists():
                print(f"🔄 reset_benchmark_baselines: removing {bdir}")
                shutil.rmtree(bdir)

    # ─── Check for existing summary.npy ─────────────────────────────────
    summary_path = Path(job.dirs["summary"]) / "summary.npy"
    print(
        f"Run job.run_init_pass?: summary_path={summary_path!r}, exists={summary_path.exists()}, overwrite={overwrite}")

    if run_init:
        if run_reference_sweep and (reference_sweep_params is not None):
            print("▶ Running parameter sweep on reference building …")

            # # Determine or generate benchmark_tag
            # if run_benchmarking and (benchmark_tag is None):
            #     now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #     benchmark_tag = f"refbench-{now}"

            # Run the sweep
            job.sweep_reference_params(
                params_to_sweep=reference_sweep_params,
                sweep_name="ref_quality_sweep",
                export_tiff=True,
                compute_metrics=True,
                all_combinations=True,
                run_benchmarking=run_benchmarking,
                benchmark_tag=benchmark_tag,
                reset_benchmark_baselines=reset_benchmark_baselines,
                profiling=profiling,
            )
            print("✔︎ Reference sweep complete.\n")
            return  # Exit early to skip init_pass and registration for this run
        else:
            if summary_path.exists() and not overwrite:
                print("✔︎ Found existing summary; loading instead of re-running init_pass()")
                job.summary = job.load_summary()
            else:
                print("➜ Running run_init_pass() …")
                t0 = time.time()
                if profiling:
                    enable_profiler()
                    set_profiler_log_file(Path(job.dirs["job_dir"]) / "profiler_log_init.txt")
                job.run_init_pass()
                if profiling:
                    disable_profiler()
                t1 = time.time()
                init_timing = t1 - t0
                if "ref_img_3d" not in job.summary:
                    raise RuntimeError("ref_img_3d not found in summary after init_pass!")
                print("✔︎ init_pass complete; ref_img_3d.shape =", job.summary["ref_img_3d"].shape)
                print(f"[TIMING] run_init_pass() took {init_timing:.2f} s")
    else:
        print("⏭ Skipping init_pass() because run_init is False")

    if run_diagnostics or run_benchmarking:

        # TODO: Is this slowing me down? Do I need to load this twice?
        job.summary = summary_dict = job.load_summary()
        out_summary_dir = Path(job.dirs["job_dir"]) / "init_summary"
        out_summary_dir.mkdir(exist_ok=True)

        # === Load and save summary contents ===
        try:

            ref3d = summary_dict.get("ref_img_3d")
            if ref3d is not None:
                np.save(out_summary_dir / "ref_img_3d.npy", ref3d)

            plane_shifts = summary_dict.get("plane_shifts") # TODO: Do we need both of these keys?
            if plane_shifts is not None:
                np.save(out_summary_dir / "plane_shifts.npy", plane_shifts)

            fuse_shift = summary_dict.get("fuse_shift")
            if fuse_shift is not None:
                np.save(out_summary_dir / "fuse_shift.npy", np.array(fuse_shift))

            fuse_shifts = summary_dict.get("fuse_shifts")
            if fuse_shifts is not None:
                np.save(out_summary_dir / "fuse_shifts.npy", fuse_shifts)

            xtalk = summary_dict.get("crosstalk_coeff")
            if xtalk is not None:
                np.save(out_summary_dir / "crosstalk_coeff.npy", xtalk)

            print(f"✔︎ Saved init_pass summary to {out_summary_dir}")
        except Exception as e:
            print(f"[WARN] Could not load/save summary dict: {e}")

    # ─── Benchmark the initialization step ────────────────────────────────

    try:
        if 'init_timing' in locals() and 'ref_img_3d' in locals():

            init_timings = {'init_pass_time_s': init_timing}

            init_outputs = {
                'ref_img': ref3d if 'ref3d' in locals() else None,
                'crosstalk_coeff': xtalk if 'xtalk' in locals() else 0.0,
                'plane_shifts': plane_shifts if 'plane_shifts' in locals() else np.zeros((num_planes, 2)),
                'fuse_shift': fuse_shift if 'fuse_shift' in locals() else (0.0, 0.0),
                'fuse_shifts': fuse_shifts if 'fuse_shifts' in locals() else np.zeros((num_planes, 2)),
            }

            step = "initialization"
            step_dir = results_dir / step
            baseline_dir = step_dir / "baseline"
            step_dir.mkdir(exist_ok=True, parents=True)

            if run_benchmarking:
                if reset_benchmark_baselines or not baseline_dir.exists():
                    save_benchmark_results(
                        step_dir,
                        outputs=init_outputs,
                        timings=init_timings,
                        repo_status=repo_status,
                        is_baseline=True
                    )
                    print(f"✔︎ initialization baseline saved to {baseline_dir}")
                else:
                    tag = benchmark_tag or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    run_dir = step_dir / tag
                    run_dir.mkdir(exist_ok=True)

                    benchmark(
                        run_dir,
                        outputs=init_outputs,
                        timings=init_timings,
                        repo_status=repo_status
                    )
                    print(f"✔︎ initialization benchmarking saved under {run_dir.name} (compared to baseline)")
            else:
                print("⏭ Skipping initialization benchmarking because run_benchmarking is False")
        else:
            print("[WARN] Skipping initialization benchmarking: missing timing or output data")
    except Exception as e:
        print(f"[WARN] initialization benchmark failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Registration: Check for existing registration chunks
    # ──────────────────────────────────────────────────────────────────────────────

    if run_register:

        reg_folder = Path(job.dirs["registered_fused_data"])
        existing_chunks = list(reg_folder.glob("fused_reg_data*.npy"))
        print(f"Run job.register?: existing_chunks count={len(existing_chunks)}, overwrite={overwrite}")

        if existing_chunks and not overwrite:
            print(f"✔︎ Found {len(existing_chunks)} existing registration chunks; skipping job.register()")
        else:
            print(f"➜ Starting registration: 3D={params['3d_reg']}, GPU={params['gpu_reg']}")
            t2 = time.time()
            # If you have large tiffs, split the large tiffs into files of size 100 after registration
            job.params['split_tif_size'] = split_tif_size
            if profiling:
                enable_profiler()   # turn on custom profiler
                set_profiler_log_file(Path(job.dirs["job_dir"]) / "profiler_log_reg.txt")
            job.register()
            if profiling:
                disable_profiler()  # turn off custom profiler
            t3 = time.time()
            reg_timing = t3 - t2
            print(f"✔︎ job.register() complete; took {reg_timing:.2f} seconds")
    else:
        print("⏭ Skipping job.register() because run_register is False")

    # ─────────────────────────────────────────────────────────────────────────────────────────

    # Create registration shifts directory
    out_shifts_dir = Path(job.dirs["job_dir"]) / "registration_shifts"
    out_shifts_dir.mkdir(exist_ok=True)

    # TODO: Remove this
    """
        START DEBUG
    """

    reg_dir = job.dirs["registered_fused_data"]
    print("\n▶ On-disk chunks in registered_fused_data:")
    for fn in sorted(os.listdir(reg_dir)):
        if fn.endswith(".npy") and "fused_reg_data" in fn:
            arr = np.load(os.path.join(reg_dir, fn), mmap_mode="r")
            print(f"  {fn:40s} → shape {arr.shape}")
    print("split_tif_size =", job.params.get("split_tif_size"))
    print("max_reg_tif_size =", job.params.get("max_reg_tif_size"))

    """
        END DEBUG
    """

    # ─── Load mov_reg if it will be used, unless it's already in memory ───

    if run_export_zstack or run_export_tiffs or run_diagnostics or run_corr_map:   # and 'mov_reg' not in locals():

        # ─── Load registered volume as Dask array ──────────────────────────
        print("\nConstructing & Exporting Registered TIFF files...")
        print("➜ Loading registered movie as a Dask array …")
        mov_reg = job.get_registered_movie(
            key="registered_fused_data",
            filename_filter="fused",
            axis=1
        )
        if mov_reg is None:
            raise RuntimeError("Could not load registered movie — aborting.")
        print("✔︎ Movie Shape:", mov_reg.shape)

        """
            START DEBUG
        """

        if debug_demo:
            print("Planes:", mov_reg.shape[0])
            print("Volumes:", mov_reg.shape[1])
            print("H, W:", mov_reg.shape[2:])
            print("params['split_tif_size']:", job.params['split_tif_size'])
            # print("params['max_reg_tif_size']:", job.params['max_reg_tif_size']) # NOTE: This makes no sense
            print("params['n_ch_tif']:", job.params['n_ch_tif'])
            print("params['n_init_files']:", job.params.get('n_init_files'))

            print("Input TIFF count:", len(job.tifs))
            print("Expected volumes:", len(job.tifs) // (job.params['n_ch_tif'] * job.params['num_colors']))
            print("\n")

        """
            END DEBUG
        """

    if run_diagnostics:

        # ─── Warn about missing inputs for diagnostics (early heads-up) ───
        required_files = [
            "mean_fused_volume.npy",
            "ymaxs_nr_blocks.npy",
            "xmaxs_nr_blocks.npy"
        ]
        for fname in required_files:
            if not (out_shifts_dir / fname).exists():
                print(f"[WARN] Missing diagnostic input: {fname}")

        if not (Path(job.dirs["job_dir"]) / "init_summary" / "summary.npy").exists():
            print("[WARN] Missing job.summary; some diagnostic plots may fail")

        # Check summary if needed for diagnostic plotting
        if run_diagnostics and not (Path(job.dirs["job_dir"]) / "init_summary" / "summary.npy").exists():
            print("[WARN] Missing job.summary; some diagnostic plots may fail")

        # === Compute + save mean fused volume & dynamic‐grid reshape ===
        try:
            # 1) Mean fused volume over time
            mean_vol = mov_reg.mean(axis=1).compute()  # (T, H, W)
            np.save(out_shifts_dir / "mean_fused_volume.npy", mean_vol)
            print(f"✔︎ Saved mean fused volume to {out_shifts_dir / 'mean_fused_volume.npy'}")

            # 2) Load and stash full offset dict
            reg_results = job.load_registration_results()
            np.savez(out_shifts_dir / "offset_dict.npz", **reg_results)

            # 3) Load chunked arrays and collapse chunk axis
            data = np.load(out_shifts_dir / "offset_dict.npz")
            y_arr = data["ymaxs_nr_blocks"]  # shape (n_chunks, T, P, nBlocks)
            x_arr = data["xmaxs_nr_blocks"]
            n_chunks, T, P, nBlocks = y_arr.shape
            total_T = n_chunks * T
            flat_y = y_arr.reshape(total_T, P, nBlocks)
            flat_x = x_arr.reshape(total_T, P, nBlocks)
            print(f"DEBUG: flat_y shape after collapse = {flat_y.shape}") if debug_demo else None

            # 4) Infer best (nYb, nXb) from nBlocks and image aspect
            H, W = mov_reg.shape[2], mov_reg.shape[3]
            aspect = H / W
            divisors = [(i, nBlocks // i) for i in range(1, int(nBlocks ** 0.5) + 1) if nBlocks % i == 0]
            nYb, nXb = min(divisors, key=lambda pair: abs((pair[0] / pair[1]) - aspect))
            print(f"DEBUG: inferred block grid = {nYb}×{nXb} (for {nBlocks} blocks)") if debug_demo else None

            # 5) Reshape into (T, P, nYb, nXb)
            blocks_y = flat_y.reshape(total_T, P, nYb, nXb)
            blocks_x = flat_x.reshape(total_T, P, nYb, nXb)
            print(f"DEBUG: blocks_y.shape = {blocks_y.shape}") if debug_demo else None

            # 6) Persist the 4-D arrays, remove previous versions
            # import os
            for fname, arr in [("ymaxs_nr_blocks.npy", blocks_y),
                               ("xmaxs_nr_blocks.npy", blocks_x)]:
                outp = out_shifts_dir / fname
                print(f"DEBUG: will write {fname} to {outp}") if debug_demo else None
                if outp.exists():
                    outp.unlink()
                    print(f"DEBUG: removed stale {outp}") if debug_demo else None
                np.save(str(outp), arr)
                print(f"DEBUG: saved new {outp} with shape {arr.shape}") if debug_demo else None

            # 7) Collapse to one shift per plane per frame (median over blocks)
            nonrigid_y = np.median(blocks_y, axis=(2, 3))  # (T, P)
            nonrigid_x = np.median(blocks_x, axis=(2, 3))
            print(f"DEBUG: nonrigid_y.shape = {nonrigid_y.shape}, nonrigid_x.shape = {nonrigid_x.shape}") if debug_demo else None
            nr_used = True

        except Exception as e:
            print(f"[WARN] Could not compute/save mean or parse offsets: {e}")
            mean_vol = np.empty((0, 0, 0))
            rigid_x = rigid_y = np.empty((0,))
            nonrigid_x = nonrigid_y = np.empty((0, 0))
            nr_used = False

        # === Extract, save, and plot fused rigid + per-plane nonrigid shifts ===
        try:
            # 1) Global rigid shifts
            reg_results = job.load_registration_results()
            int_shift = np.array(reg_results["int_shift"][0])  # (T, 3)
            rigid_y = int_shift[:, 1]
            rigid_x = int_shift[:, 2]

            # 2) Reload the canonical offsets .npz
            np.savez(out_shifts_dir / "offset_dict.npz", **reg_results)
            data = np.load(out_shifts_dir / "offset_dict.npz")
            y_arr = data["ymaxs_nr_blocks"]  # (n_chunks, T, P, nBlocks)
            x_arr = data["xmaxs_nr_blocks"]

            # 3) Collapse chunk → frame axis
            n_chunks, T, P, nBlocks = y_arr.shape
            total_T = n_chunks * T
            flat_y = y_arr.reshape(total_T, P, nBlocks)
            flat_x = x_arr.reshape(total_T, P, nBlocks)
            print(f"DEBUG: flat_y.shape = {flat_y.shape}") if debug_demo else None

            # 4) Infer (nYb, nXb) by factoring nBlocks to match image aspect
            H, W = mov_reg.shape[2], mov_reg.shape[3]
            aspect = H / W
            pairs = [(i, nBlocks // i) for i in range(1, int(nBlocks ** 0.5) + 1) if nBlocks % i == 0]
            nYb, nXb = min(pairs, key=lambda p: abs((p[0] / p[1]) - aspect))
            print(f"DEBUG: inferred block grid = {nYb}×{nXb} for {nBlocks} blocks") if debug_demo else None

            # 5) Reshape into full 4-D grid
            blocks_y = flat_y.reshape(total_T, P, nYb, nXb)
            blocks_x = flat_x.reshape(total_T, P, nYb, nXb)
            print(f"DEBUG: blocks_y.shape = {blocks_y.shape}") if debug_demo else None

            # 6) Compute per-plane medians
            nonrigid_y = np.median(blocks_y, axis=(2, 3))  # → (T, P)
            nonrigid_x = np.median(blocks_x, axis=(2, 3))
            print(f"DEBUG: nonrigid_y.shape = {nonrigid_y.shape}, nonrigid_x.shape = {nonrigid_x.shape}") if debug_demo else None
            nr_used = True

        except Exception as e:
            print(f"[WARN] Could not load or parse registration shifts: {e}")
            rigid_x = rigid_y = np.empty((0,))
            nonrigid_x = nonrigid_y = np.empty((0, 0))
            nr_used = False

        # === Plot global rigid + per-plane nonrigid shifts ===
        try:
            # === Plot global rigid + per-plane nonrigid shifts ===
            try:

                print("✔︎ Plotting registration shift diagnostics...")
                print("  - rigid_x shape:", rigid_x.shape)
                print("  - rigid_y shape:", rigid_y.shape)

                # Ensure shape is (num_planes, num_volumes) → average over planes
                if rigid_x.ndim == 2:
                    rigid_x_plot = rigid_x.mean(axis=0)
                    rigid_y_plot = rigid_y.mean(axis=0)
                    x_vals = np.arange(rigid_x.shape[1])
                elif rigid_x.ndim == 1:
                    rigid_x_plot = rigid_x
                    rigid_y_plot = rigid_y
                    x_vals = np.arange(len(rigid_x))
                else:
                    raise ValueError(f"Unexpected shape for rigid_x: {rigid_x.shape}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x_vals, rigid_x_plot, label="Rigid X", alpha=0.8)
                ax.plot(x_vals, rigid_y_plot, label="Rigid Y", alpha=0.8)
                ax.set_title("Global Rigid Shifts Over Time")
                ax.set_xlabel("Volume Index")
                ax.set_ylabel("Shift (pixels)")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()

                # Now the per-plane nonrigid plot
                fig, ax = plt.subplots(figsize=(8, 4))
                for p in range(nonrigid_x.shape[1]):
                    ax.plot(nonrigid_x[:, p], "--", label=f"Plane {p} Nonrigid X", alpha=0.6)
                    ax.plot(nonrigid_y[:, p], "--", label=f"Plane {p} Nonrigid Y", alpha=0.6)
                ax.set_title("Per-Plane Nonrigid Shifts Over Time")
                ax.set_xlabel("Volume Index")
                ax.set_ylabel("Shift (pixels)")
                ax.legend(ncol=2, fontsize="small")
                ax.grid(True)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"[WARN] Could not plot registration shifts: {e}")

            # fig, ax = plt.subplots(figsize=(8, 4))
            # ax.plot(rigid_x, label="Rigid X", alpha=0.8)
            # ax.plot(rigid_y, label="Rigid Y", alpha=0.8)
            # ax.set_title("Global Rigid Shifts Over Time")
            # ax.set_xlabel("Volume Index")
            # ax.set_ylabel("Shift (pixels)")
            # ax.legend()
            # ax.grid(True)
            # plt.tight_layout()
            # plt.show()
            #
            # fig, ax = plt.subplots(figsize=(8, 4))
            # for p in range(nonrigid_x.shape[1]):
            #     ax.plot(nonrigid_x[:, p], "--", label=f"Plane {p} Nonrigid X", alpha=0.6)
            #     ax.plot(nonrigid_y[:, p], "--", label=f"Plane {p} Nonrigid Y", alpha=0.6)
            # ax.set_title("Per-Plane Nonrigid Shifts Over Time")
            # ax.set_xlabel("Volume Index")
            # ax.set_ylabel("Shift (pixels)")
            # ax.legend(ncol=2, fontsize="small")
            # ax.grid(True)
            # plt.tight_layout()
            # plt.show()

        except Exception as e:
            print(f"[WARN] Could not plot registration shifts: {e}")

        # ─── Nonrigid block‐wise shifts vs. time ─────────────────────────────────
        try:
            # 1) Load the fully-shaped block arrays written in Step 6
            blocks_y = np.load(out_shifts_dir / "ymaxs_nr_blocks.npy")  # (T, P, nYb, nXb)
            blocks_x = np.load(out_shifts_dir / "xmaxs_nr_blocks.npy")
            T, P, nYb, nXb = blocks_y.shape
            print(f"DEBUG: plotting nonrigid blocks from shape {blocks_y.shape}") if debug_demo else None

            # 2) Plot every block as a faint line
            fig, (axY, axX) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axY.set_title("Nonrigid blockwise Y-shifts over time")
            axX.set_title("Nonrigid blockwise X-shifts over time")
            axX.set_xlabel("Frame")
            axY.set_ylabel("Shift (px)")
            axX.set_ylabel("Shift (px)")

            for p in range(P):
                # Flatten the (nYb, nXb) grid into nYb*nXb time series
                flatY = blocks_y[:, p].reshape(T, -1)  # → (T, nYb*nXb)
                flatX = blocks_x[:, p].reshape(T, -1)
                for b in range(flatY.shape[1]):
                    axY.plot(flatY[:, b], alpha=0.2)
                    axX.plot(flatX[:, b], alpha=0.2)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[WARN] Could not plot nonrigid shifts vs time: {e}")

        # ─── Plane-alignment offsets (from init_pass) ────────────────────────────
        try:
            # 1) Grab the init_pass summary (reuse job.summary if it exists)
            summary_dict = getattr(job, "summary", None) or job.load_summary()

            # 2) Check for plane_shifts
            if "plane_shifts" not in summary_dict:
                raise KeyError("plane_shifts not found in init_pass summary")

            plane_shifts = np.array(summary_dict["plane_shifts"])  # shape (P, 2)
            print(f"DEBUG: plane_shifts.shape = {plane_shifts.shape}") if debug_demo else None

            # 3) Extract Y and X offsets
            ys, xs = plane_shifts[:, 0], plane_shifts[:, 1]
            P = plane_shifts.shape[0]

            # 4) Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(P), ys, "-o", label="Plane Y-offset")
            ax.plot(np.arange(P), xs, "-s", label="Plane X-offset")
            ax.set_title("Init-pass Plane Alignment Offsets")
            ax.set_xlabel("Plane index")
            ax.set_ylabel("Offset (pixels)")
            ax.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[WARN] Could not plot plane alignment offsets: {e}")

        # ─── Mean 2D vector‐field overlay ─────────────────────────
        try:
            # 1) Load the mean fused volume and nonrigid shifts
            mean_vol = np.load(out_shifts_dir / "mean_fused_volume.npy")  # (P, H, W)
            shifts_y = np.load(out_shifts_dir / "ymaxs_nr_blocks.npy")  # (T, P, nYb, nXb)
            shifts_x = np.load(out_shifts_dir / "xmaxs_nr_blocks.npy")
            T, P, nYb, nXb = shifts_y.shape

            plane = 1  # or whichever plane you'd like to show
            if plane >= P:
                raise ValueError(f"Plane {plane} out of range (P={P})")

            # 2) Compute the per‐tile mean shifts over time
            mean_dy = shifts_y[:, plane].mean(axis=0)  # → (nYb, nXb)
            mean_dx = shifts_x[:, plane].mean(axis=0)

            # 3) Retrieve the exact block edges & form the 2D grid of centers
            summary = job.load_summary()
            raw_yedges = np.array(summary["reference_params"]["yblock"])  # (nYb*nXb, 2)
            raw_xedges = np.array(summary["reference_params"]["xblock"])

            y_edges = raw_yedges.reshape(nYb, nXb, 2)  # → (nYb, nXb, [start,end])
            x_edges = raw_xedges.reshape(nYb, nXb, 2)

            tile_ctr_y = y_edges.mean(axis=2)  # → (nYb, nXb)
            tile_ctr_x = x_edges.mean(axis=2)

            # 4) Scale arrows for visibility
            scale_factor = 15.0
            dx_vis = mean_dx * scale_factor
            dy_vis = mean_dy * scale_factor

            # 5) Plot overlay
            P_img, H, W = mean_vol.shape[0], mean_vol.shape[1], mean_vol.shape[2]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(
                mean_vol[plane],
                cmap="gray",
                origin="upper",
                extent=[0, W, H, 0]  # ensure Y starts at top
            )
            q = ax.quiver(
                tile_ctr_x, tile_ctr_y,
                dx_vis, dy_vis,
                angles="xy",
                scale_units="xy",
                scale=1,  # we already scaled
                color="r",
                width=0.005
            )

            # 6) Clamp axes to the image
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal")

            ax.set_title(f"Mean Nonrigid Warp Vector‐Field (plane {plane}, scale×{scale_factor})")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[WARN] Could not plot mean vector‐field overlay: {e}")

        # ─── Export per‐plane vector‐field over time as GIF ─────────────────────────
        if generate_nonrigid_vector_field_gifs: # NOTE: Flag set at the top of the script
            try:

                arrow_scale_mult = 0.05  # how “long” the arrows are relative to their raw magnitude
                arrow_width = 0.005  # thickness of the arrows

                mean_vol = np.load(out_shifts_dir / "mean_fused_volume.npy")  # (P, H, W)
                shifts_y = np.load(out_shifts_dir / "ymaxs_nr_blocks.npy")  # (T, P, nYb, nXb)
                shifts_x = np.load(out_shifts_dir / "xmaxs_nr_blocks.npy")
                T, P, nYb, nXb = shifts_y.shape
                H, W = mean_vol.shape[1], mean_vol.shape[2]
                by, bx = params["block_size"]

                # precompute your quiver grid
                y_centers = (np.arange(nYb) + 0.5) * by
                x_centers = (np.arange(nXb) + 0.5) * bx
                Xc, Yc = np.meshgrid(x_centers, y_centers)

                for p in range(P):
                    frames = []
                    for t in range(T):
                        dx = shifts_x[t, p]
                        dy = shifts_y[t, p]

                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(
                            mean_vol[p],
                            cmap="gray",
                            origin="upper",
                            extent=[0, W, H, 0],
                        )

                        # auto‐compute a base scale so arrows fill ~half your tile
                        m = max(np.abs(dx).max(), np.abs(dy).max(), 1e-3)
                        scale = m * arrow_scale_mult

                        ax.quiver(
                            Xc, Yc, dx, dy,
                            angles="xy",
                            scale_units="xy",
                            scale=scale,
                            width=arrow_width,
                            color="red",
                            pivot="mid",
                        )

                        ax.set_xlim(0, W)
                        ax.set_ylim(H, 0)
                        ax.set_aspect("equal")
                        ax.axis("off")

                        # render via Agg → RGB array
                        canvas = FigureCanvas(fig)
                        canvas.draw()
                        raw, (w, h) = canvas.print_to_buffer()
                        img = (
                            np.frombuffer(raw, dtype=np.uint8)
                            .reshape(h, w, 4)[..., :3]
                        )

                        frames.append(img)
                        plt.close(fig)

                    out_gif = out_shifts_dir / f"vector_field_over_time_plane{p}.gif"
                    imageio.mimsave(str(out_gif), frames, fps=gif_fps)
                    print(f"✔︎ Saved GIF for plane {p} at {gif_fps} fps → {out_gif}")

            except Exception as e:
                print(f"[WARN] Could not export per-plane vector-field GIFs: {e}")

        # ─── First vs. last vector‐field comparison (Pick two time points) ─────────────────
        try:
            # 1) Load mean fused volume and blockwise nonrigid shifts
            mean_vol = np.load(out_shifts_dir / "mean_fused_volume.npy")  # shape (P, H, W)
            blocks_y = np.load(out_shifts_dir / "ymaxs_nr_blocks.npy")  # shape (T, P, nYb, nXb)
            blocks_x = np.load(out_shifts_dir / "xmaxs_nr_blocks.npy")
            T, P, nYb, nXb = blocks_y.shape

            # 2) Compute tile‐center coordinates from block_size
            by, bx = params["block_size"]
            y_centers = (np.arange(nYb) + 0.5) * by
            x_centers = (np.arange(nXb) + 0.5) * bx
            Xc, Yc = np.meshgrid(x_centers, y_centers)

            # 3) Choose which plane to visualize
            plane = 1
            if not (0 <= plane < P):
                raise ValueError(f"Plane {plane} out of range (got P={P})")

            # 4) Grab the first and last time‐point shifts for that plane
            dy_all = blocks_y[:, plane]  # (T, nYb, nXb)
            dx_all = blocks_x[:, plane]

            # compute per‐frame mean magnitude on the given plane
            mag = np.sqrt(
                dx_all.mean(axis=(1, 2)) ** 2 +
                dy_all.mean(axis=(1, 2)) ** 2
            )  # shape (T,)

            # find the indices of the two largest magnitudes
            # 1) argsort gives ascending order, 2) reverse and take first two
            sorted_idx_desc = np.argsort(mag)[::-1]
            t_peaks = sorted_idx_desc[:2]
            t0, t1 = t_peaks

            # Grab the first and last frames
            # t0, t1 = 0, T - 1
            # t0, t1 = 50, T - 50
            dy0, dx0 = dy_all[t0], dx_all[t0]
            dy1, dx1 = dy_all[t1], dx_all[t1]

            # 5) Auto‐compute a scale so arrows are visible but not giant
            max_shift = max(
                np.abs(dx0).max(), np.abs(dy0).max(),
                np.abs(dx1).max(), np.abs(dy1).max(),
                1e-3
            )
            # arrow_length_desired = how many pixels you want your max_shift arrow to span
            arrow_length = 20
            scale = max_shift / arrow_length
            # scale = 0.1

            # 6) Plot side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
            for ax, (dx, dy, t) in zip(axes[0], [(dx0, dy0, t0), (dx1, dy1, t1)]):
                ax.imshow(mean_vol[plane], cmap="gray", origin="upper")
                ax.quiver(
                    Xc, Yc, dx, dy,
                    angles="xy", scale_units="xy", scale=scale,
                    width=0.005, color="red", pivot="mid"
                )
                ax.set_title(f"Vector-field at frame {t} (plane {plane})")
                ax.set_xlabel("X (px)")
                ax.set_ylabel("Y (px)")
                ax.set_xlim(0, mean_vol.shape[2])
                ax.set_ylim(mean_vol.shape[1], 0)  # invert Y to match image coords
                ax.set_aspect("equal")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[WARN] Could not plot first vs. last vector fields: {e}")

        # ──────────────────────────────────────────────────────────────────────────────

    else:
        print("⏭ Skipping diagnostics, plots, and shift export because run_diagnostics is False")
        # mov_reg = None  # important: make mov_reg available to later blocks (TIFF export or corr_map)

    # ─── Benchmark the registration step ────────────────────────────────

    try:
        if 'rigid_x' in locals() and 'rigid_y' in locals():
            reg_timings = {'registration_time_s': reg_timing if 'reg_timing' in locals() else 0.0}
            reg_outputs = {
                'rigid_x': rigid_x,
                'rigid_y': rigid_y,
                'nonrigid_x': nonrigid_x if nonrigid_x.size else np.zeros((rigid_x.size, 0)),
                'nonrigid_y': nonrigid_y if nonrigid_y.size else np.zeros((rigid_y.size, 0)),
                'mean_fused_volume': mean_vol if 'mean_vol' in locals() else None,
            }

            step = "registration"
            step_dir = results_dir / step
            baseline_dir = step_dir / "baseline"

            step_dir.mkdir(exist_ok=True, parents=True)

            if run_benchmarking:
                if reset_benchmark_baselines or not baseline_dir.exists():
                    save_benchmark_results(
                        step_dir,
                        outputs=reg_outputs,
                        timings=reg_timings,
                        repo_status=repo_status,
                        is_baseline=True
                    )
                    print(f"✔︎ registration baseline saved to {baseline_dir}")
                else:
                    # Create unique tag subdir: either user tag or timestamp
                    tag = benchmark_tag or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    run_dir = step_dir / tag
                    run_dir.mkdir(exist_ok=True)

                    benchmark(
                        run_dir,
                        outputs=reg_outputs,
                        timings=reg_timings,
                        repo_status=repo_status
                    )
                    print(f"✔︎ registration benchmarking saved under {run_dir.name} (compared to baseline)")
            else:
                print("⏭ Skipping registration benchmarking because run_benchmarking is False")
        else:
            print("[WARN] Skipping registration benchmarking: no rigid‐shift data")
    except Exception as e:
        print(f"[WARN] registration benchmark failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    # 4. (Optional) Compute + save correlation map on mov_reg
    # ──────────────────────────────────────────────────────────────────────────────

    if compute_corr_map:
        # 1) Compute and save
        try:
            t_corr_start = time.time()
            # returns (vmap, mean_img, max_img), roi_masks, neuropil_masks
            (vmap, mean_img, max_img), _, _ = job.calculate_corr_map(mov=mov_reg)
            corr_timing = time.time() - t_corr_start
            print(f"[TIMING] correlation map computation took {corr_timing:.2f} s")

            out_corr_dir = Path(job.dirs["job_dir"]) / "corr_map"
            out_corr_dir.mkdir(exist_ok=True)
            np.save(out_corr_dir / "vmap.npy", vmap)
            np.save(out_corr_dir / "mean_img.npy", mean_img)
            np.save(out_corr_dir / "max_img.npy", max_img)
            print(f"✔︎ Saved correlation map outputs to {out_corr_dir}")
        except Exception as e:
            print(f"[WARN] Could not compute or save correlation map: {e}")


        # ─── Benchmark the correlation step ────────────────────────────────

        try:
            if 'corr_timing' in locals():
                corr_timings = {'corr_map_time_s': corr_timing}
                corr_outputs = {
                    'vmap': vmap,
                    'mean_img': mean_img,
                    'max_img': max_img,
                }

                step = "corr_map"
                step_dir = results_dir / step
                baseline_dir = step_dir / "baseline"
                step_dir.mkdir(exist_ok=True, parents=True)

                if run_benchmarking:
                    if reset_benchmark_baselines or not baseline_dir.exists():
                        save_benchmark_results(
                            step_dir,
                            outputs=corr_outputs,
                            timings=corr_timings,
                            repo_status=repo_status,
                            is_baseline=True
                        )
                        print(f"✔︎ corr_map baseline saved to {baseline_dir}")
                    else:
                        tag = benchmark_tag or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        run_dir = step_dir / tag
                        run_dir.mkdir(exist_ok=True)

                        benchmark(
                            run_dir,
                            outputs=corr_outputs,
                            timings=corr_timings,
                            repo_status=repo_status
                        )
                        print(f"✔︎ corr_map benchmarking saved under {run_dir.name} (compared to baseline)")
                else:
                    print("⏭ Skipping corr_map benchmarking because run_benchmarking is False")
            else:
                print("[WARN] Skipping corr_map benchmarking: no timing data")
        except Exception as e:
            print(f"[WARN] corr_map benchmark failed: {e}")

        # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────
    # 5. Write TIFF Movie Files To Disk
    # ──────────────────────────────────────────────────────────────────────────────

    # # ─── Safety check: mov_reg must exist to write TIFFs ───
    # if run_export_tiffs and 'mov_reg' not in locals():
    #     raise RuntimeError("mov_reg is required for TIFF export but was not generated or loaded.")

    # ─── Generate ZStack Projections (prior to TIFF export) ───────────────

    if run_export_zstack:
        print("▶ Generating registered zstack projections …")
        t4 = time.time()
        # NOTE: mov_reg is returned as persist but chunked for generating z-stacks
        #   mov_reg = mov_reg.rechunk({0: -1})
        #   Axis 0 is Z, and this sets it to a single chunk across Z
        zstack_dict, mov_reg = job.generate_registered_zstack(
            methods=zstack_projection_types,
            mov_reg=None,  # will load and persist if not yet done
            planes=None,  # or specify subset if needed
            save_npy=True,
            save_tif=True,
            verbose=True
        )
        t5 = time.time()
        print(f"[TIMING] TIFF zstack writing took {t5 - t4:.2f} seconds")
    else:
        print("▶ Skipping zstack projection …")
        # mov_reg = job.get_registered_movie()
        # if mov_reg is not None:
        #     mov_reg = mov_reg.persist()

    # ─── Write per‐plane, chunked TIFFs in parallel ───────────────────────────────

    if run_export_tiffs_by_planes:
        try:
            print("➜ Rechunking Movie …")
            mov_reg = safe_rechunk_for_tiff_export(mov_reg)
            print("➜ Writing registered TIFFs …")
            # Write under “extended‐length” parent: extended_root + “/registered_tiffs”
            out_tiff_dir = Path(job.dirs["job_dir"]) / "registered_tiffs"
            print("num_tiff_export_workers: {}".format(num_tiff_export_workers))

            t6 = time.time()
            print("\n📦 Starting Dask `persist()` to materialize rechunked movie …")
            print(f"   ⊳ Shape: {mov_reg.shape}")
            print(f"   ⊳ Chunks: {{Z: {mov_reg.chunks[0]}, T: {mov_reg.chunks[1]}}}")
            print("   ⊳ This may take a while if the dataset is large.\n")
            start_persist_time = time.time()
            with diagnostics.ProgressBar():
                mov_reg = mov_reg.persist()
            total_persist_time = time.time() - start_persist_time
            print("\n✅ Dask `persist()` complete.")
            print(f"   ⊳ Elapsed Time: {total_persist_time:.2f} seconds")
            print(f"   ⊳ Persisted Type: {type(mov_reg)}")
            print(f"   ⊳ Persisted Chunks: {{Z: {mov_reg.chunks[0]}, T: {mov_reg.chunks[1]}}}")
            print(f"   ⊳ Stored on: {mov_reg.__dask_keys__()[0][:2]} ... (truncated)\n")

            write_registered_planes_parallel(
                mov_reg,
                output_dir=str(out_tiff_dir),
                max_tiff_pages=max_reg_tif_size,
                max_workers=num_tiff_export_workers,
            )
            t7 = time.time()
            print(f"[TIMING] TIFF writing by plane took {t7 - t6:.2f} seconds")
        except Exception as e:
            print(f"❌ TIFF export failed: {e}")
    else:
        print("⏭ Skipping TIFF by plane export because run_export_tiffs_by_planes is False")

    t_end = time.time()
    print(f"[TIMING] Total script time: {t_end - t_start:.2f} seconds")


if __name__ == "__main__":
    main()
    print("\n✅ All steps complete.")
