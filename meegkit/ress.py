"""Rhythmic Entrainment Source Separation."""
import numpy as np
from scipy import linalg

from .utils import demean, gaussfilt, theshapeof, tscov, mrdivide


def RESS(X, sfreq: int, peak_freq: float, neig_freq: float = 1,
         neig_width: float = 1, n_keep: int = 1,
         show: bool = False):
    """Rhythmic entrainment source separation [1]_.

    Parameters
    ----------
    X: array, shape=(n_samples, n_chans, n_trials)
        Data to denoise.
    sfreq : int
        Sampling frequency.
    peak_freq : float
        Peak frequency.
    neig_freq : float
        Distance of neighbouring frequencies away from peak frequency, in Hz.
    neig_width : float
        FWHM of the neighboring frequencies (default=1).
    n_keep : int
        Number of components to keep.

    References
    ----------
    .. [1] Cohen, M. X., & Gulbinaite, R. (2017). Rhythmic entrainment source
       separation: Optimizing analyses of neural responses to rhythmic sensory
       stimulation. Neuroimage, 147, 43-56.

    """
    n_samples, n_chans, n_trials = theshapeof(X)
    X = demean(X)

    # Covariance of signal and covariance of noise
    c01, _ = tscov(gaussfilt(X, sfreq, peak_freq + neig_freq,
                             fwhm=neig_width, n_harm=1))
    c02, _ = tscov(gaussfilt(X, sfreq, peak_freq - neig_freq,
                             fwhm=1, n_harm=1))
    c1, _ = tscov(gaussfilt(X, sfreq, peak_freq, fwhm=1, n_harm=1))

    # perform generalized eigendecomposition
    d, V = linalg.eig(c1, (c01 + c02) / 2)
    d = d.real
    V = V.real

    # Sort eigenvectors by decreasing eigenvalues
    idx = np.argsort(d)[::-1]
    d = d[idx]
    V = V[:, idx]

    # Truncate weak components
    # if thresh is not None:
    #     idx = np.where(d / d.max() > thresh)[0]
    #     d = d[idx]
    #     V = V[:, idx]

    # Keep a fixed number of components
    # max_comps = np.min((n_keep, V.shape[1]))
    # V = V[:, np.arange(max_comps)]
    # d = d[np.arange(max_comps)]

    # Normalize components
    V /= np.sqrt(np.sum(V, axis=0) ** 2)

    # extract components and force sign
    maps = mrdivide(c1 @ V, V.T @ c1 @ V)
    idx = np.argmax(np.abs(maps[:, 0]))  # find biggest component
    maps = maps * np.sign(maps[idx, 0])  # force to positive sign

    # reconstruct RESS component time series
    out = np.zeros((n_samples, n_keep, n_trials))
    for t in range(n_trials):
        out[..., t] = X[:, :, t] @ V[:, np.arange(n_keep)]

    if n_keep == 1:
        out = out.squeeze(1)

    return out
