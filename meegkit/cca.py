"""CCA functions."""
import sys
import numpy as np
from scipy import linalg

from .utils.covariances import cov_lags
from .utils.matrix import _check_shifts, normcol, relshift, _times_to_delays
from .utils.denoise import pca


def cca_crossvalidate(xx, yy, shifts=None, sfreq=1, surrogate=False,
                      plot=False):
    """CCA with cross-validation.

    Parameters
    ----------
    xx : list of arrays
        If a list is provided, each element should have shape = (n_times,
        n_chans). If array, it should be 3D of shape = (n_times, n_chans,
        n_trials).
    yy : list of arrays
        If a list is provided, each element should have shape = (n_times,
        n_chans). If array, it should be 3D of shape = (n_times, n_chans,
        n_trials).
    shifts : array, shape = (n_shifts,)
        Array of shifts to apply to y relative to x (can be negative).
    surrogate : bool
        If True, estimate sd of correlation over non-matching pairs.
    plot : bool
        Produce some plots.

    Returns
    -------
    AA, BB : arrays
        Cell arrays of transform matrices.
    RR : array, shape = (n_comps, n_shifts, n_trials)
        Correlations (2D).
    SD : array
        Standard deviation of correlation over non-matching pairs (2D).

    """
    if isinstance(xx, list) and isinstance(yy, list):
        assert len(xx) == len(yy)
    elif isinstance(xx, np.ndarray) and isinstance(yy, np.ndarray):
        assert xx.shape[-1] == yy.shape[-1]
        # Convert xx and yy to lists
        xx = [xx[..., t] for t in np.arange(xx.shape[-1])]
        yy = [yy[..., t] for t in np.arange(yy.shape[-1])]
    else:
        raise AttributeError('xx and yy both must be lists of same length, ' +
                             'or arrays os same n_trials.')

    # Calculate covariance matrices
    print('Calculate all covariances...')
    shifts = _times_to_delays(shifts, sfreq)
    shifts, n_shifts = _check_shifts(shifts)
    n_trials = len(xx)
    n_feats = xx[0].shape[1] + yy[0].shape[1]  # sum of channels

    C = np.zeros((n_feats, n_feats, n_shifts, n_trials)).squeeze()
    for iTrial in range(n_trials):
        C[..., iTrial], _, _ = cov_lags(xx[iTrial], yy[iTrial], shifts)

    # Calculate leave-one-out CCAs
    print('Calculate CCAs...')
    AA = list()
    BB = list()
    for t in range(n_trials):
        # covariance of all trials except t
        CC = np.sum(C[..., np.arange(n_trials) != t], axis=-1, keepdims=True)
        if CC.ndim == 4:
            CC = np.squeeze(CC, 3)

        # corresponding CCA
        [A, B, R] = nt_cca(None, None, None, CC, xx[0].shape[1])
        AA.append(A)
        BB.append(B)

    del C, CC

    # Calculate leave-one-out correlation coefficients
    sys.stdout.write('Calculate cross-correlations...')
    sys.stdout.flush()
    n_comps = A.shape[1]
    r = np.zeros((n_comps, n_shifts))
    RR = np.zeros((n_comps, n_shifts, n_trials))
    for t in range(n_trials):
        A = AA[t]
        B = BB[t]
        for s in range(n_shifts):
            x, y = relshift(xx[t], yy[t], shifts[s])
            a = A[:, :, s]
            b = B[:, :, s]
            r[:, s] = np.diag(np.dot(normcol(np.dot(x, a)).T,
                                     normcol(np.dot(y, b)))) / x.shape[0]
        RR[:, :, t] = r

        # if surrogate:
        #     ss = np.zeros_like(RR)
        #     next = np.mod(t, n_trials)  # correlate with next in list
        #     for s in np.arange(n_shifts):
        #         [x, y] = relshift(xx[t], yy[next], shifts[s])
        #         a = A[:, :, s]
        #         b = B[:, :, s]
        #         m = np.min((x.shape[0], y.shape[0]))
        #         s[:, s] = np.diag(np.dot(normcol(np.dot(x, a)).T,
        #                                  normcol(np.dot(y, b)))) / m
        #     ss[:, :, t] = s

    sys.stdout.write('\b OK!\n')

    # if surrogate:
    #     var = (np.sum(ss ** 2, 2) - np.sum(ss, 2) ** 2 / n_trials) /\
    #           (n_trials - 1)
    #     SD = np.sqrt(var)

    if plot:
        import matplotlib.pyplot as plt
        f, (ax1) = plt.subplots(1, 1)
        for k in range(RR.shape[0]):
            ax1.plot(shifts, np.mean(RR[k, :, :], 1).T, label='CC{}'.format(k))
        ax1.set_title('correlation for each CC')
        ax1.set_xlabel('shift')
        ax1.set_ylabel('correlation')
        ax1.legend()

        # if surrogate:
        #     ax1.plot(SD.T, ':')

        f2, axes = plt.subplots(min(4, RR.shape[0]), 1)
        for k, ax in zip(np.arange(min(4, RR.shape[0])), axes):
            idx = np.argmax(np.mean(RR[k, :, :], 1))
            [x, y] = relshift(xx[0], yy[0], shifts[idx])
            ax.plot(np.dot(x, A[:, k, idx]).T, label='CC{}'.format(k))
            ax.plot(np.dot(y, B[:, k, idx]).T, ':')
            ax.legend()

        ax.set_xlabel('sample')
        f2.set_tight_layout(True)
        plt.show()

    return AA, BB, RR


def nt_cca(X=None, Y=None, lags=None, C=None, m=None, thresh=None, sfreq=1):
    """Compute CCA from covariance.

    Parameters
    ----------
    X, Y : arrays, shape = (n_times, n_chans[, n_trials])
        Data.
    lags : array, shape = (n_lags,)
        Array of lags. A positive lag means Y delayed relative to X. If
        :attr:`sfreq` is > 1, lags are interpreted as times in seconds.
    C : array, shape = (n_chans, n_chans[, n_lags])
        Covariance matrix of [X, Y]. C can be 3D, which case CCA is derived
        independently from each page.
    m : int
        Number of channels of X.
    thresh: float
        Discard principal components below this value.
    sfreq : float
        Sampling frequency. If not 1, lags are assumed to be given in seconds.

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
        lags = _times_to_delays(lags, sfreq)
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
    Cxw = whiten_nt(C[:m, :m], thresh)  # sphere X
    Cyw = whiten_nt(C[m:, m:], thresh)  # sphere Y

    # apply sphering matrices to C
    W = np.zeros((Cxw.shape[0] + Cyw.shape[0], Cxw.shape[1] + Cyw.shape[1]))
    W[:Cxw.shape[0], :Cxw.shape[1]] = Cxw
    W[Cxw.shape[0]:, Cxw.shape[1]:] = Cyw
    C = np.dot(np.dot(W.T, C), W)

    # Number of CCA componenets
    N = np.min((Cxw.shape[1], Cyw.shape[1]))

    # PCA
    V, d = pca(C)

    A = np.dot(Cxw, V[:Cxw.shape[1], :N]) * np.sqrt(2)
    B = np.dot(Cyw, V[Cxw.shape[1]:, :N]) * np.sqrt(2)
    R = d[:N] - 1

    return A, B, R


def whiten(C, fudge=1e-18):
    """Whiten covariance matrix C of X.

    If X has shape = (observations, components), X_white = np.dot(X, W).

    References
    ----------
    https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca

    """
    eigvals, V = linalg.eigh(C)  # eigenvalue decomposition of the covariance

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(eigvals + fudge))
    W = np.dot(np.dot(V, D), V.T)   # whitening matrix

    return W


def whiten_nt(C, thresh=1e-12):
    """Covariance whitening function from noisetools."""
    d, V = linalg.eigh(C)  # eigh if matrix symmetric, eig otherwise
    d = np.real(d)
    V = np.real(V)

    # Sort eigenvalues
    idx = np.argsort(d)[::-1]
    d = d[idx]
    V = V[:, idx]

    # Remove small eigenvalues
    keep = (d / np.max(d)) > thresh
    d = d[keep]
    V = V[:, keep]

    # break symmetry when x and y perfectly correlated (otherwise cols of x*A
    # and y*B are not orthogonal)
    d = d ** (1 - thresh)

    D = np.diag(np.sqrt((1. / d)))
    W = np.dot(V, D)

    return W


def whiten_svd(X):
    """SVD whitening."""
    U, S, Vt = linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white


def whiten_zca(C, thresh=None):
    """Compute ZCA whitening matrix (aka Mahalanobis whitening).

    Parameters
    ----------
    C : array
        Covariance matrix.
    thresh : float
        Whitening constant, it prevents division by zero.

    Returns
    -------
    ZCA: array, shape (n_chans, n_chans)
        ZCA matrix, to be multiplied with data.

    """
    U, S, V = np.linalg.svd(C)  # Singular Value Decomposition

    # ZCA Whitening matrix
    D = np.diag(1. / np.sqrt(S + thresh))
    ZCA = np.dot(np.dot(U, D), U.T)

    return ZCA
