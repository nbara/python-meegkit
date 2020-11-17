"""Rhythmic Entrainment Source Separation."""
import numpy as np
from scipy import linalg

from .utils import demean, gaussfilt, theshapeof, tscov, mrdivide


def RESS(X, sfreq: int, peak_freq: float, neig_freq: float = 1,
         peak_width: float = .5, neig_width: float = 1, n_keep: int = 1,
         return_maps: bool = False, show: bool = False):
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
        Distance of neighbouring frequencies away from peak frequency, +/- in
        Hz (default=1).
    peak_width : float
        FWHM of the peak frequency (default=.5).
    neig_width : float
        FWHM of the neighboring frequencies (default=1).
    n_keep : int
        Number of components to keep (default=1). -1 keeps all components.
    return_maps : bool
        If True, also output maps (mixing matrix).

    Returns
    -------
    out : array, shape=(n_samples, n_keep, n_trials)
        RESS time series.
    maps : array, shape=(n_channels, n_keep)
        If return_maps is True, also output mixing matrix.

    Notes
    -----
    To project the RESS components back into sensor space, one can proceed as
    follows. First apply RESS:
    >> out, maps = ress.RESS(data, sfreq, peak_freq, return_maps=True)

    Then multiply each trial by the mixing matrix:
    >> from meegkit.utils import matmul3d
    >> proj = matmul3d(out, maps.T)

    References
    ----------
    .. [1] Cohen, M. X., & Gulbinaite, R. (2017). Rhythmic entrainment source
       separation: Optimizing analyses of neural responses to rhythmic sensory
       stimulation. Neuroimage, 147, 43-56.

    """
    n_samples, n_chans, n_trials = theshapeof(X)
    X = demean(X)

    if n_keep == -1:
        n_keep = n_chans

    # Covariance of signal and covariance of noise
    c01, _ = tscov(gaussfilt(X, sfreq, peak_freq + neig_freq,
                             fwhm=neig_width, n_harm=1))
    c02, _ = tscov(gaussfilt(X, sfreq, peak_freq - neig_freq,
                             fwhm=neig_width, n_harm=1))
    c1, _ = tscov(gaussfilt(X, sfreq, peak_freq, fwhm=peak_width, n_harm=1))

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

    # Normalize components
    V /= np.sqrt(np.sum(V, axis=0) ** 2)

    # extract components
    maps = mrdivide(c1 @ V, V.T @ c1 @ V)
    maps = maps[:, :n_keep]
    # idx = np.argmax(np.abs(maps[:, 0]))  # find biggest component
    # maps = maps * np.sign(maps[idx, 0])  # force to positive sign

    # reconstruct RESS component time series
    out = np.zeros((n_samples, n_keep, n_trials))
    for t in range(n_trials):
        out[..., t] = X[:, :, t] @ V[:, np.arange(n_keep)]

    if return_maps:
        return out, maps
    else:
        return out
