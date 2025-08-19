import numpy as np
from skimage.metrics import structural_similarity, normalized_mutual_information
from scipy.ndimage import laplace

def compute_ncc(ref: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    Normalized cross‐correlation of each frame against the reference.
    ref: 2D (H×W), frames: 3D (T×H×W) → returns (T,)
    """
    r0 = ref - ref.mean()
    σr = r0.std()
    out = []
    for f in frames:
        f0 = f - f.mean()
        σf = f0.std()
        corr = float((r0.ravel() @ f0.ravel()) / (σr * σf * f0.size))
        out.append(corr)
    return np.array(out)


def compute_ssim(ref: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    Structural Similarity Index (SSIM) per frame.
    """
    scores = []
    for f in frames:
        score = structural_similarity(ref, f, data_range=f.max() - f.min())
        scores.append(score)
    return np.array(scores)


def compute_mse(ref: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    Pixel‐wise mean squared error per frame.
    """
    return np.mean((frames - ref)**2, axis=(1, 2))


def compute_rmse(ref: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.sqrt(compute_mse(ref, frames))


def compute_residual_motion(frames: np.ndarray) -> np.ndarray:
    """
    Frame‐to‐frame RMS difference: sqrt(mean((f[t]−f[t−1])²)).
    Returns length T−1.
    """
    diffs = frames[1:] - frames[:-1]
    return np.sqrt((diffs**2).mean(axis=(1, 2)))


def compute_mutual_information(
    ref: np.ndarray,
    frames: np.ndarray,
    bins: int = 64
) -> np.ndarray:
    """
    Normalized mutual information between ref and each frame.
    """
    out = []
    rflat = ref.ravel()
    for f in frames:
        fflat = f.ravel()
        mi = normalized_mutual_information(rflat, fflat, bins=bins)
        out.append(mi)
    return np.array(out)


def compute_sharpness_fraction(frame: np.ndarray, thresh: float) -> float:
    """
    Fraction of pixels whose Laplacian magnitude > thresh.
    A basic sharpness proxy.
    """
    L = np.abs(laplace(frame))
    return float((L > thresh).sum() / L.size)


def summarize_metrics(
    ref: np.ndarray,
    orig: np.ndarray,
    reg: np.ndarray,
    sharp_thresh: float = None
):
    """
    Compute all blind quality metrics before vs. after registration.

    Returns:
      - results: dict of per‐frame arrays
      - summaries: dict of scalar summaries for benchmarking
    """
    # per‐frame
    ncc_pre  = compute_ncc(ref, orig)
    ncc_post = compute_ncc(ref, reg)

    ssim_pre  = compute_ssim(ref, orig)
    ssim_post = compute_ssim(ref, reg)

    mse_pre   = compute_mse(ref, orig)
    mse_post  = compute_mse(ref, reg)

    resid_post = compute_residual_motion(reg)

    mi_pre    = compute_mutual_information(ref, orig)
    mi_post   = compute_mutual_information(ref, reg)

    results = {
        'ncc_pre': ncc_pre, 'ncc_post': ncc_post,
        'ssim_pre': ssim_pre, 'ssim_post': ssim_post,
        'mse_pre': mse_pre, 'mse_post': mse_post,
        'resid_post': resid_post,
        'mi_pre': mi_pre, 'mi_post': mi_post,
    }

    # scalar summaries
    summaries = {
        'mean_ncc_improvement': float(ncc_post.mean() - ncc_pre.mean()),
        'median_ssim_post':    float(np.median(ssim_post)),
        'rmse_reduction':      float(np.mean(np.sqrt(mse_pre)) - np.mean(np.sqrt(mse_post))),
        'mean_residual_motion': float(resid_post.mean()),
    }

    if sharp_thresh is not None:
        sharp_fracs = [compute_sharpness_fraction(f, sharp_thresh) for f in reg]
        results['sharp_frac_post'] = np.array(sharp_fracs)
        summaries['mean_sharp_frac_post'] = float(np.mean(sharp_fracs))

    return results, summaries
