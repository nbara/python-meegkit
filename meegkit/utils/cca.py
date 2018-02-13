import numpy as np
from scipy import linalg

from .covariances import cov_lags
from .matrix import _check_shifts


def nt_cca(X=None, Y=None, lags=None, C=None, m=None, thresh=None):
    """Compute CCA from covariance.

    Parameters
    ----------
    X, Y : arrays, shape = (n_times, n_chans[, n_trials])
        Data.
    lags : array, shape = (n_lags,)
        Array of lags. A positive lag means Y delayed relative to X.
    C : array, shape = (n_chans, n_chans[, n_lags])
        Covariance matrix of [X, Y]. C can be 3D, which case CCA is derived
        independently from each page.
    m : int
        Number of channels of X.
    thresh: float
        Discard principal components below this value.

    Returns
    -------
    A : array, shape = (n_chans_X, min(n_chans_X, n_chans_Y))
        Transform matrix mapping `X` to canonical space.
    B : array,  shape = (n_chans_Y, min(n_chans_X, n_chans_Y))
        Transform matrix mapping `Y` to canonical space.
    R : array, shape = (n_comps, n_lags)
        Correlation scores.

    Notes
    -----
    Usage 1: CCA of X, Y
    >> [A, B, R] = nt_cca(X, Y)  # noqa

    Usage 2: CCA of X, Y for each value of lags.
    >> [A, B, R] = nt_cca(X, Y, lags)  # noqa

    A positive lag indicates that Y is delayed relative to X.

    Usage 3: CCA from covariance matrix
    >> C = [X, Y].T * [X, Y]  # noqa
    >> [A, B, R] = nt_cca([], [], [], C, X.shape[1])  # noqa

    Use the third form to handle multiple files or large data (covariance C can
    be calculated chunk-by-chunk).

    .. warning:: Means of X and Y are NOT removed.
    .. warning:: A, B are scaled so that (X * A)^2 and (Y * B)^2 are identity
                 matrices (differs from sklearn).

    See Also
    --------
    nt_cov_lags, nt_relshift, nt_cov, nt_pca in NoiseTools.

    """
    if thresh is None:
        thresh = 1e-12

    if (X is None and Y is not None) or (Y is None and X is not None):
        raise AttributeError('Either *both* X and Y should be defined, or C!')

    if X is not None:
        lags, n_lags = _check_shifts(lags)
        C, _, m = cov_lags(X, Y, lags)
        A, B, R = nt_cca(None, None, None, C, m, thresh)
        return A, B, R

    if C is None:
        raise RuntimeError('covariance matrix should be defined')
    if m is None:
        raise RuntimeError('m should be defined')
    if C.shape[0] != C.shape[1]:
        raise RuntimeError('covariance matrix should be square')
    if any((X, Y, lags)):
        raise RuntimeError('only covariance should be defined at this point')
    if C.ndim > 3:
        raise RuntimeError('covariance should be 3D at most')

    if C.ndim == 3:  # covariance is 3D: do a separate CCA for each trial
        n_chans, _, n_lags = C.shape
        N = np.min((m, n_chans - m))
        A = np.zeros((m, N, n_lags))
        B = np.zeros((n_chans - m, N, n_lags))
        R = np.zeros((N, n_lags))

        for k in np.arange(n_lags):
            AA, BB, RR = nt_cca(None, None, None, C[:, :, k], m, thresh)
            A[:AA.shape[0], :AA.shape[1], k] = AA
            B[:BB.shape[0], :BB.shape[1], k] = BB
            R[:, k] = RR

        return A, B, R

    # Calculate CCA given C = [X,Y].T * [X,Y] and m = x.shape[1]
    # -------------------------------------------------------------------------
    Cxw = nt_whiten(C[:m, :m], thresh)  # sphere X
    Cyw = nt_whiten(C[m:, m:], thresh)  # sphere Y

    # apply sphering matrices to C
    W = np.zeros((Cxw.shape[0] + Cyw.shape[0], Cxw.shape[1] + Cyw.shape[1]))
    W[:Cxw.shape[0], :Cxw.shape[1]] = Cxw
    W[Cxw.shape[0]:, Cxw.shape[1]:] = Cyw
    C = np.dot(np.dot(W.T, C), W)

    # Number of CCA componenets
    N = np.min((Cxw.shape[1], Cyw.shape[1]))

    # PCA
    d, V = linalg.eig(C)
    d, V = np.real(d), np.real(V)
    idx = np.argsort(d)[::-1]
    d, V = d[idx], V[:, idx]
    A = np.dot(Cxw, V[:Cxw.shape[1], :N]) * np.sqrt(2)
    B = np.dot(Cyw, V[Cxw.shape[1]:, :N]) * np.sqrt(2)
    R = d[:N] - 1

    return A, B, R


def whiten(C, fudge=1e-18):
    """Whiten covariance matrix C of X.

    If X should has shape = (observations, components), X_white = np.dot(X, W).

    References
    ----------
    https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca

    """
    d, V = linalg.eigh(C)  # eigenvalue decomposition of the covariance

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))
    W = np.dot(np.dot(V, D), V.T)   # whitening matrix

    return W


def nt_whiten(C, thresh=1e-12):
    """Whiten function from noisetools."""
    d, V = linalg.eig(C)
    d, V = np.real(d), np.real(V)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    keep = (d / np.max(d)) > thresh
    V = V[:, idx[keep]]
    d = d[keep]
    d = d ** (1 - thresh)
    Cw = np.dot(V, np.diag(np.sqrt((1. / d))).T)

    return Cw


def svd_whiten(X):
    """SVD whitening."""
    U, s, Vt = linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white
