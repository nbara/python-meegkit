"""Rhythmic Entrainment Source Separation."""
import numpy as np
from scipy import linalg

from .utils import demean, gaussfilt, theshapeof, tscov, mrdivide


def RESS(X, sfreq: int, peak_freq: float, neig_freq: float = 1,
         peak_width: float = .5, neig_width: float = 1, n_keep: int = 1,
         return_maps: bool = False):
    """Rhythmic Entrainment Source Separation.

    As described in [1]_.

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
        If True, also output mixing (to_ress) and unmixing matrices
        (from_ress), used to transform the data into RESS component space and
        back into sensor space, respectively.

    Returns
    -------
    out : array, shape=(n_samples, n_keep, n_trials)
        RESS time series.
    from_ress : array, shape=(n_components, n_channels)
        Unmixing matrix (projects to sensor space).
    to_ress : array, shape=(n_channels, n_components)
        Mixing matrix (projects to component space).

    Examples
    --------
    To project the RESS components back into sensor space, one can proceed as
    follows:

    >>> # First apply RESS
    >>> from meegkit.utils import matmul3d  # handles 3D matrix multiplication
    >>> out, fromRESS, _ = ress.RESS(data, sfreq, peak_freq, return_maps=True)
    >>> # Then matrix multiply each trial by the unmixing matrix:
    >>> proj = matmul3d(out, fromRESS)

    To transform a new observation into RESS component space (e.g. in the
    context of a cross-validation, with separate train/test sets):

    >>> # Start by applying RESS to the train set:
    >>> out, _, toRESS = ress.RESS(data, sfreq, peak_freq, return_maps=True)
    >>> # Then multiply your test data by the toRESS:
    >>> new_comp = new_data @ toRESS

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
    d, to_ress = linalg.eig(c1, (c01 + c02) / 2)
    d = d.real
    to_ress = to_ress.real

    # Sort eigenvectors by decreasing eigenvalues
    idx = np.argsort(d)[::-1]
    d = d[idx]
    to_ress = to_ress[:, idx]

    # Truncate weak components
    # if thresh is not None:
    #     idx = np.where(d / d.max() > thresh)[0]
    #     d = d[idx]
    #     to_ress = to_ress[:, idx]

    # Normalize components (yields mixing matrix)
    to_ress /= np.sqrt(np.sum(to_ress, axis=0) ** 2)
    to_ress = to_ress[:, np.arange(n_keep)]

    # Compute unmixing matrix
    from_ress = mrdivide(c1 @ to_ress, to_ress.T @ c1 @ to_ress).T
    from_ress = from_ress[:n_keep, :]

    # idx = np.argmax(np.abs(from_ress[:, 0]))  # find biggest component
    # from_ress = from_ress * np.sign(from_ress[idx, 0])  # force positive sign

    # Output `n_keep` RESS component time series
    out = np.zeros((n_samples, n_keep, n_trials))
    for t in range(n_trials):
        out[..., t] = X[:, :, t] @ to_ress

    if return_maps:
        return out, from_ress, to_ress
    else:
        return out
