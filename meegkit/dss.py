"""Denoising source separation."""
# Authors:  Nicolas Barascud <nicolas.barascud@gmail.com>
#           Maciej Szul <maciej.szul@isc.cnrs.fr>
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import linalg
from scipy.signal import welch

from .tspca import tsr
from .utils import (
    demean,
    gaussfilt,
    matmul3d,
    mean_over_trials,
    pca,
    smooth,
    theshapeof,
    tscov,
    wpwr,
)


def dss1(X, weights=None, keep1=None, keep2=1e-12):
    """DSS to maximise repeatability across trials.

    Evoked-biased DSS denoising.

    Parameters
    ----------
    X: array, shape=(n_samples, n_chans, n_trials)
        Data to denoise.
    weights: array
        Weights.
    keep1: int
        Number of PCs to retain in function:`dss0` (default=all).
    keep2: float
        Ignore PCs smaller than keep2 in function:`dss0` (default=1e-12).

    Returns
    -------
    todss: array, shape=(n_dss_components, n_chans)
        Denoising matrix to convert X to normalized DSS components.
    from: array, shape=(n_dss_components, n_chans)
        Matrix to convert DSS components back to sensor space.
    pwr0: array
        Power per component (raw).
    pwr1: array
        Power per component (averaged).

    """
    n_trials = theshapeof(X)[-1]

    # if demean: # remove weighted mean
    #   X = demean(X, weights)

    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(X, weights)
    ww /= n_trials

    # covariance of raw and biased X
    c0, nc0 = tscov(X, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c0 /= nc0
    c1 /= nc1

    todss, fromdss, pwr0, pwr1 = dss0(c0, c1, keep1, keep2)

    return todss, fromdss, pwr0, pwr1


def dss0(c0, c1, keep1=None, keep2=1e-9, return_unmixing=True):
    """DSS base function.

    This function allows specifying arbitrary bias functions (as compared to
    the function:`dss1`, which forces the bias to be the mean over trials).

    Parameters
    ----------
    c0: array, shape=(n_chans, n_chans)
        Baseline covariance.
    c1: array, shape=(n_chans, n_chans)
        Biased covariance.
    keep1: int | None
        Number of PCs to retain (default=None, which keeps all).
    keep2: float
        Ignore PCs smaller than keep2 (default=1e-9).
    return_unmixing : bool
        If True (default), return the unmixing matrix.

    Returns
    -------
    todss: array, shape=(n_chans, n_dss_components)
        Matrix to convert X to normalized DSS components.
    fromdss : array, shape=(n_dss_components, n_chans)
        Matrix to transform back to original space. Only returned if
        ``return_unmixing`` is True.
    pwr0: array
        Power per component (baseline).
    pwr1: array
        Power per component (biased).

    Notes
    -----
    The data mean is NOT removed prior to processing.

    """
    # Check size and squareness
    assert c0.shape == c1.shape == (c0.shape[0], c0.shape[0]), \
        "c0 and c1 should have the same size, and be square"

    # Check for NaN or INF
    assert not (np.any(np.isnan(c0)) or np.any(np.isinf(c0))), "NaN or INF in c0"
    assert not (np.any(np.isnan(c1)) or np.any(np.isinf(c1))), "NaN or INF in c1"

    # derive PCA and whitening matrix from unbiased covariance
    eigvec0, eigval0 = pca(c0, max_comps=keep1, thresh=keep2)

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    W = np.diag(np.sqrt(1. / eigval0))  # diagonal of whitening matrix

    # c1 is projected into whitened PCA space of data channels
    c2 = (eigvec0 @ W).T @ c1 @ (eigvec0 @ W)

    # proj. matrix from whitened data space to a space maximizing bias
    eigvec2, eigval2 = pca(c2, max_comps=keep1, thresh=keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = eigvec0 @ W @ eigvec2

    # Normalise DSS matrix
    N = np.sqrt(np.diag(todss.T @ c0 @ todss))
    todss /= N

    pwr0 = np.sqrt(np.sum((c0 @ todss) ** 2, axis=0))
    pwr1 = np.sqrt(np.sum((c1 @ todss) ** 2, axis=0))

    # Return data
    # next line equiv. to: np.array([np.dot(todss, ep) for ep in data])
    # dss_data = np.einsum('ij,hjk->hik', todss, data)

    if return_unmixing:
        fromdss = linalg.pinv(todss)
        return todss, fromdss, pwr0, pwr1
    else:
        return todss, pwr0, pwr1


def dss_line(X, fline, sfreq, nremove=1, nfft=1024, nkeep=None, blocksize=None,
             show=False):
    """Apply DSS to remove power line artifacts.

    Implements the ZapLine algorithm described in [1]_.

    Parameters
    ----------
    X : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency (normalized to sfreq, if ``sfreq`` == 1).
    sfreq : float
        Sampling frequency (default=1, which assymes ``fline`` is normalised).
    nremove : int
        Number of line noise components to remove (default=1).
    nfft : int
        FFT size (default=1024).
    nkeep : int
        Number of components to keep in DSS (default=None).
    blocksize : int
        If not None (default), covariance is computed on blocks of
        ``blocksize`` samples. This may improve performance for large datasets.
    show: bool
        If True, show DSS results (default=False).

    Returns
    -------
    y : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    artifact : array, shape=(n_samples, n_chans, n_trials)
        Artifact

    Examples
    --------
    Apply to X, assuming line frequency=50Hz and sampling rate=1000Hz, plot
    results:
    >>> dss_line(X, 50/1000)

    Removing 4 line-dominated components:
    >>> dss_line(X, 50/1000, 4)

    Truncating PCs beyond the 30th to avoid overfitting:
    >>> dss_line(X, 50/1000, 4, nkeep=30);

    Return cleaned data in y, noise in yy, do not plot:
    >>> [y, artifact] = dss_line(X, 60/1000)

    References
    ----------
    .. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to
       remove power line artifacts [Preprint]. https://doi.org/10.1101/782029

    """
    if X.shape[0] < nfft:
        print(f"Reducing nfft to {X.shape[0]}")
        nfft = X.shape[0]
    n_samples, n_chans, _ = theshapeof(X)
    if blocksize is None:
        blocksize = n_samples

    # Recentre data
    X = demean(X, inplace=True)

    # Cancel line_frequency and harmonics + light lowpass
    X_filt = smooth(X, sfreq / fline)

    # X - X_filt results in the artifact plus some residual biological signal
    X_noise = X - X_filt

    # Reduce dimensionality to avoid overfitting
    if nkeep is not None:
        cov_X_res = tscov(X_noise)[0]
        V, _ = pca(cov_X_res, nkeep)
        X_noise_pca = X_noise @ V
    else:
        X_noise_pca = X_noise.copy()
        nkeep = n_chans

    # Compute blockwise covariances of raw and biased data
    n_harm = np.floor((sfreq / 2) / fline).astype(int)
    c0 = np.zeros((nkeep, nkeep))
    c1 = np.zeros((nkeep, nkeep))
    for X_block in sliding_window_view(X_noise_pca, (blocksize, nkeep),
                                       axis=(0, 1))[::blocksize, 0]:
        # if n_trials>1, reshape to (n_samples, nkeep, n_trials)
        if X_block.ndim == 3:
            X_block = X_block.transpose(1, 2, 0)

        # bias data
        c0 += tscov(X_block)[0]
        c1 += tscov(gaussfilt(X_block, sfreq, fline, fwhm=1, n_harm=n_harm))[0]

    # DSS to isolate line components from residual
    todss, _, pwr0, pwr1 = dss0(c0, c1)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(pwr1 / pwr0, ".-")
        plt.xlabel("component")
        plt.ylabel("score")
        plt.title("DSS to enhance line frequencies")
        plt.show()

    # Remove line components from X_noise
    idx_remove = np.arange(nremove)
    X_artifact = matmul3d(X_noise_pca, todss[:, idx_remove])
    X_res = tsr(X_noise, X_artifact)[0]  # project them out

    # reconstruct clean signal
    y = X_filt + X_res

    # Power of components
    p = wpwr(X - y)[0] / wpwr(X)[0]
    print(f"Power of components removed by DSS: {p:.2f}")
    # return the reconstructed clean signal, and the artifact
    return y, X - y


def dss_line_iter(data, fline, sfreq, win_sz=10, spot_sz=2.5,
                  nfft=512, show=False, prefix="dss_iter", n_iter_max=100):
    """Remove power line artifact iteratively.

    This method applies dss_line() until the artifact has been smoothed out
    from the spectrum.

    Parameters
    ----------
    data : data, shape=(n_samples, n_chans, n_trials)
        Input data.
    fline : float
        Line frequency.
    sfreq : float
        Sampling frequency.
    win_sz : float
        Half of the width of the window around the target frequency used to fit
        the polynomial (default=10).
    spot_sz : float
        Half of the width of the window around the target frequency used to
        remove the peak and interpolate (default=2.5).
    nfft : int
        FFT size for the internal PSD calculation (default=512).
    show: bool
        Produce a visual output of each iteration (default=False).
    prefix : str
        Path and first part of the visualisation output file
        "{prefix}_{iteration number}.png" (default="dss_iter").
    n_iter_max : int
        Maximum number of iterations (default=100).

    Returns
    -------
    data : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    iterations : int
        Number of iterations.
    """

    def nan_basic_interp(array):
        """Nan interpolation."""
        nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
        array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
        return array

    freq_rn = [fline - win_sz, fline + win_sz]
    freq_sp = [fline - spot_sz, fline + spot_sz]
    freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)

    freq_rn_ix = np.logical_and(freq >= freq_rn[0], freq <= freq_rn[1])
    freq_used = freq[freq_rn_ix]
    freq_sp_ix = np.logical_and(freq_used >= freq_sp[0],
                                freq_used <= freq_sp[1])

    if psd.ndim == 3:
        mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
    elif psd.ndim == 2:
        mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

    mean_psd_wospot = mean_psd.copy()
    mean_psd_wospot[freq_sp_ix] = np.nan
    mean_psd_tf = nan_basic_interp(mean_psd_wospot)
    pf = np.polyfit(freq_used, mean_psd_tf, 3)
    p = np.poly1d(pf)
    clean_fit_line = p(freq_used)

    aggr_resid = []
    iterations = 0
    while iterations < n_iter_max:
        data, _ = dss_line(data, fline, sfreq, nfft=nfft, nremove=1)
        freq, psd = welch(data, fs=sfreq, nfft=nfft, axis=0)
        if psd.ndim == 3:
            mean_psd = np.mean(psd, axis=(1, 2))[freq_rn_ix]
        elif psd.ndim == 2:
            mean_psd = np.mean(psd, axis=(1))[freq_rn_ix]

        residuals = mean_psd - clean_fit_line
        mean_score = np.mean(residuals[freq_sp_ix])
        aggr_resid.append(mean_score)

        print(f"Iteration {iterations} score: {mean_score}")

        if show:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2, 2, figsize=(12, 6), facecolor="white")

            if psd.ndim == 3:
                mean_sens = np.mean(psd, axis=2)
            elif psd.ndim == 2:
                mean_sens = psd

            y = mean_sens[freq_rn_ix]
            ax.flat[0].plot(freq_used, y)
            ax.flat[0].set_title("Mean PSD across trials")

            ax.flat[1].plot(freq_used, mean_psd_tf, c="gray")
            ax.flat[1].plot(freq_used, mean_psd, c="blue")
            ax.flat[1].plot(freq_used, clean_fit_line, c="red")
            ax.flat[1].set_title("Mean PSD across trials and sensors")

            tf_ix = np.where(freq_used <= fline)[0][-1]
            ax.flat[2].plot(residuals, freq_used)
            color = "green"
            if mean_score <= 0:
                color = "red"
            ax.flat[2].scatter(residuals[tf_ix], freq_used[tf_ix], c=color)
            ax.flat[2].set_title("Residuals")

            ax.flat[3].plot(np.arange(iterations + 1), aggr_resid, marker="o")
            ax.flat[3].set_title("Iterations")

            plt.tight_layout()
            plt.savefig(f"{prefix}_{iterations:03}.png")
            plt.close("all")

        if mean_score <= 0:
            break

        iterations += 1

    if iterations == n_iter_max:
        raise RuntimeError("Could not converge. Consider increasing the "
                           "maximum number of iterations")

    return data, iterations
