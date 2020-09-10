"""Denoising source separation."""
import numpy as np
from scipy import linalg

from .tspca import tsr
from .utils import (demean, gaussfilt, mean_over_trials, pca, smooth,
                    theshapeof, tscov, wpwr)


def dss1(data, weights=None, keep1=None, keep2=1e-12):
    """DSS to maximise repeatability across trials.

    Evoked-biased DSS denoising.

    Parameters
    ----------
    data: array, shape=(n_samples, n_chans, n_trials)
        Data to denoise.
    weights: array
        Weights.
    keep1: int
        Number of PCs to retain in function:`dss0` (default=all).
    keep2: float
        Ignore PCs smaller than keep2 in function:`dss0` (default=10^-12).

    Returns
    -------
    todss: array, shape=(n_dss_components, n_chans)
        Ddenoising matrix to convert data to normalized DSS components.
    pwr0: array
        Power per component (raw).
    pwr1: array
        Power per component (averaged).

    Notes
    -----
    The data mean is NOT removed prior to processing.

    """
    n_samples, n_chans, n_trials = theshapeof(data)
    data = demean(data, weights)  # remove weighted mean

    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(data, weights)

    ww = ww.min(1)

    # covariance of raw and biased data
    c0, nc0 = tscov(data, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c1 = c1 / n_trials

    todss, fromdss, ratio, pwr = dss0(c0, c1, keep1, keep2)

    return todss, fromdss, ratio, pwr


def dss0(c0, c1, keep1=None, keep2=1e-9):
    """DSS base function.

    This function allows specifying arbitrary bias functions (as compared to
    the function:`dss1`, which forces the bias to be the mean over trials).

    Parameters
    ----------
    c0: array, shape=(n_chans, n_chans)
        Baseline covariance.
    c1: array, shape=(n_chans, n_chans)
        Biased covariance.
    keep1: int
        Number of PCs to retain (default=all).
    keep2: float
        Ignore PCs smaller than keep2 (default=10.^-9).

    Returns
    -------
    todss: array, shape=(n_dss_components, n_chans)
        Matrix to convert data to normalized DSS components.
    pwr0: array
        Power per component (baseline).
    pwr1: array
        Power per component (biased).

    """
    if c0 is None or c1 is None:
        raise AttributeError('dss0 needs at least two arguments')
    if c0.shape != c1.shape:
        raise AttributeError('c0 and c1 should have same size')
    if c0.shape[0] != c0.shape[1]:
        raise AttributeError('c0 should be square')
    if np.any(np.isnan(c0)) or np.any(np.isinf(c0)):
        raise ValueError('NaN or INF in c0')
    if np.any(np.isnan(c1)) or np.any(np.isinf(c1)):
        raise ValueError('NaN or INF in c1')

    # derive PCA and whitening matrix from unbiased covariance
    eigvec0, eigval0 = pca(c0, max_comps=keep1, thresh=keep2)

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    W = np.sqrt(1. / eigval0)  # diagonal of whitening matrix

    # c1 is projected into whitened PCA space of data channels
    c2 = (W * eigvec0).T.dot(c1).dot(eigvec0) * W

    # proj. matrix from whitened data space to a space maximizing bias
    eigvec2, eigval2 = pca(c2, max_comps=keep1, thresh=keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = (W[np.newaxis, :] * eigvec0).dot(eigvec2)
    fromdss = linalg.pinv(todss)

    # Normalise DSS matrix
    N = np.sqrt(1. / np.diag(np.dot(np.dot(todss.T, c0), todss)))
    todss = todss * N

    pwr0 = np.sqrt(np.sum(np.dot(c0, todss) ** 2, axis=0))
    pwr1 = np.sqrt(np.sum(np.dot(c1, todss) ** 2, axis=0))

    # Return data
    # next line equiv. to: np.array([np.dot(todss, ep) for ep in data])
    # dss_data = np.einsum('ij,hjk->hik', todss, data)

    return todss, fromdss, pwr0, pwr1


def dss_line(x, fline, sfreq, nremove=1, nfft=1024, nkeep=None, show=False):
    """Apply DSS to remove power line artifacts.

    Implements the ZapLine algorithm described in [1]_.

    Parameters
    ----------
    x : data, shape=(n_samples, n_chans, n_trials)
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

    Returns
    -------
    y : array, shape=(n_samples, n_chans, n_trials)
        Denoised data.
    artifact : array, shape=(n_samples, n_chans, n_trials)
        Artifact

    Examples
    --------
    Apply to x, assuming line frequency=50Hz and sampling rate=1000Hz, plot
    results:
    >>> dss_line(x, 50/1000)

    Removing 4 line-dominated components:
    >>> dss_line(x, 50/1000, 4)

    Truncating PCs beyond the 30th to avoid overfitting:
    >>> dss_line(x, 50/1000, 4, nkeep=30);

    Return cleaned data in y, noise in yy, do not plot:
    >>> [y, artifact] = dss_line(x, 60/1000)

    References
    ----------
    .. [1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to
       remove power line artifacts [Preprint]. https://doi.org/10.1101/782029

    """
    if x.shape[0] < nfft:
        print('reducing nfft to {}'.format(x.shape[0]))
        nfft = x.shape[0]
    n_samples, n_chans, n_trials = theshapeof(x)
    x = demean(x)

    # cancels line_frequency and harmonics, light lowpass
    xx = smooth(x, sfreq / fline)

    # residual (x=xx+xxx), contains line and some high frequency power
    xxx = x - xx

    # reduce dimensionality to avoid overfitting
    if nkeep is not None:
        xxx_cov = tscov(xxx)[0]
        V, _ = pca(xxx_cov, nkeep)
        xxxx = xxx * V
    else:
        xxxx = xxx.copy()

    # DSS to isolate line components from residual:
    n_harm = np.floor((sfreq / 2) / fline).astype(int)
    c0, _ = tscov(xxxx)
    c1, _ = tscov(gaussfilt(xxxx, sfreq, fline, 1, n_harm=n_harm))

    todss, _, pwr0, pwr1 = dss0(c0, c1)

    if show:
        import matplotlib.pyplot as plt
        plt.plot(pwr1 / pwr0, '.-')
        plt.xlabel('component')
        plt.ylabel('score')
        plt.title('DSS to enhance line frequencies')
        plt.show()

    idx_remove = np.arange(nremove)
    if x.ndim == 3:
        for t in range(n_trials):  # line-dominated components
            xxxx[..., t] = xxxx[..., t] @ todss[:, idx_remove]
    elif x.ndim == 2:
        xxxx = xxxx @ todss[:, idx_remove]

    xxx, _, _, _ = tsr(xxx, xxxx)  # project them out

    # reconstruct clean signal
    y = xx + xxx
    artifact = x - y

    # Power of components
    p = wpwr(x - y)[0] / wpwr(x)[0]
    print('Power of components removed by DSS: {:.2f}'.format(p))
    return y, artifact
