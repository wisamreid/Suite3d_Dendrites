import numpy as n
from scipy.stats import entropy, kurtosis
from sklearn.decomposition import PCA


def volume_quality(volume, pct_high = 99.99, pct_low = 25.0):
    '''
    return some heuristics about the signal level in the mean volume provided

    Args:
        volume (ndarray): nz,ny,nx - averaged movie over time
        pct_high (float, optional): high percentile to take as the peak of 'signal' . Defaults to 99.99.
        pct_low (float, optional): low percentile to take as 'background'. Defaults to 25.0.

    Returns:
        metrics: dictionary
    '''
    sig = n.percentile(volume, pct_high, axis=(1,2))
    bg = n.percentile(volume, pct_low, axis=(1,2))


    metrics = {
        'signal_range':  sig - bg,
        'signal_to_background_ratio' : sig / bg,
        'mean_fluorescence' : volume.mean(axis=(1,2)),
        'volume_std'  : volume.std(axis=(1,2)),
    }
    return metrics

def shot_noise_pct(fs, frate_hz):
    '''
    compute the theoretical shot noise percentage in a timeseries 
    assumes GCamP6s, using equation from Pachitariu

    Args:
        fs (ndarray): npix, nt - timeseries
        frate_hz (float): frame rate

    Returns:
        noise_level: array of percentage noise for each pixel
    '''
    df = n.diff(fs, axis=1)
    dff = df / fs.mean(axis=1,keepdims=True)
    abs_d_dff = n.abs(n.diff(dff,axis=1))
    noise_level = n.nanmedian(abs_d_dff, axis=1)
    noise_level = noise_level / frate_hz

    return noise_level

def choose_top_pix(vol, pct = 98):
    nz, ny, nx = vol.shape
    pcts = n.array([n.percentile(vol[i].flatten(), pct)  for i in range(nz)])
    n_top_pix = int((100-pct) * ny * nx / 100 - 2)
    top_pix = n.array([vol[i] >= pcts[i] for i in range(nz)])
    return top_pix

def compute_metrics_for_movie(mov, frate_hz, top_pix=None):
    nz, nt, ny, nx = mov.shape
    vol = mov.mean(axis=1)
    metrics = volume_quality(vol)
    if top_pix is None:
        top_pix = choose_top_pix(vol)

    noises = []
    npix = (top_pix.sum(axis=(1,2))).min()
    # print(npix)
    for i in range(nz):
        noises.append(shot_noise_pct(mov[i][:,top_pix[i]][:,:npix].reshape(nt,-1).T, frate_hz))
        # print(noises[-1].shape)
    noise_levels = n.array(noises)

    metrics['noise_levels'] = noise_levels

    return vol, metrics


def compute_reference_quality_metrics(volume: n.ndarray, debug: bool = False) -> dict:
    """
    Compute blind quality metrics on a 3D reference volume.

    Args:
        volume (np.ndarray): Reference volume of shape (Z, Y, X)
        debug (bool): If True, print debug info

    Returns:
        dict: Dictionary of metrics (keys map to per-plane arrays or scalar summaries)
    """
    metrics = volume_quality(volume)
    nz, ny, nx = volume.shape

    # Histogram entropy and kurtosis
    entropies = []
    kurtoses = []
    for i in range(nz):
        hist, _ = n.histogram(volume[i], bins=256, density=True)
        hist += 1e-12  # Avoid log(0)
        entropies.append(entropy(hist))
        kurtoses.append(kurtosis(volume[i].ravel(), fisher=False))

    metrics['histogram_entropy'] = n.array(entropies)
    metrics['histogram_kurtosis'] = n.array(kurtoses)

    # PCA z-blur metric
    flat = volume.reshape(nz, -1)
    pca = PCA(n_components=1)
    pca.fit(flat.T)
    explained_var = pca.explained_variance_ratio_[0]
    metrics['pca_explained_variance_ratio'] = explained_var

    if debug:
        print("Volume shape:", volume.shape)
        for k, v in metrics.items():
            if isinstance(v, n.ndarray):
                print(f"{k}: mean={v.mean():.4f}, std={v.std():.4f}")
            else:
                print(f"{k}: {v:.4f}")

    return metrics

    
