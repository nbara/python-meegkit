"""Sensor noise suppression."""
import numpy as np

from .tspca import tsr
from .utils import demean, fold, pca, theshapeof, tscov, unfold
from .utils.matrix import _check_weights


def sns(X, n_neighbors=0, skip=0, weights=np.array([])):
    """Sensor Noise Suppresion.

    This algorithm will replace the data from each channel by its regression on
    the subspace formed by the other channels. The underlying assumptions are
    that (a) sensor noise is uncorrelated across sensors, and (b) genuine
    signal is correlated, sensor noise is removed and genuine signal is
    preserved.

    Parameters
    ----------
    X : array, shape=(n_times, n_chans, n_trials)
        EEG data.
    n_neighbors : int
        Number of neighbors (based on correlation) to include in the
        projection.
    skip: int
        Number of closest neighbors to skip (default=0).
    weights : array of floats
        Weights (default=all ones).


    Returns
    -------
    y : array, shape=(n_times, n_chans, n_trials)
        Denoised data.
    r : array, shape=(n_chans, n_chans)
        Denoising matrix.

    """
    if not n_neighbors:
        n_neighbors = X.shape[1] - 1
    ndims = X.ndim
    weights = _check_weights(weights, X)
    n_samples, n_chans, n_trials = theshapeof(X)

    X = unfold(X)
    X = demean(X)
    c, nc = tscov(X)

    if weights.any():
        weights = unfold(weights)
        X = demean(X, weights)
        wc, nwc = tscov(X, shifts=None, weights=weights)
        r = sns0(c, n_neighbors, skip, wc)
    else:
        r = sns0(c, n_neighbors, skip, c)

    y = np.dot(np.squeeze(X), r)
    if ndims > 2:
        y = fold(y, n_samples)

    return y, r


def sns0(c, n_neighbors=0, skip=0, wc=np.array([])):
    """Sensor Noise Suppresion from data covariance.

    Parameters
    ----------
    c: array, shape=(n_chans, n_chans)
        Full covariance of data to denoise.
    n_neighbors : int
        Number of neighbors (based on correlation) to include in the
        projection.
    skip: int
        Number of closest neighbors to skip (default=0).
    wc: array
        Weighted covariance.

    Returns
    -------
    r: arraym
        Denoising matrix.

    """
    if not wc.any():
        wc = c.copy()

    n_chans = c.shape[0]

    if n_neighbors == 0:
        n_neighbors = n_chans - skip - 1
    else:
        n_neighbors = np.min((n_neighbors, n_chans - skip - 1))

    r = np.zeros(c.shape)

    # normalize
    eps = np.finfo(np.float32).eps
    norm = np.diag(c).copy()
    mask = norm > eps
    norm[mask] = np.sqrt(1. / norm[mask])
    norm[~mask] = 0
    c = c * norm
    c *= norm[:, np.newaxis]
    del norm, mask

    for k in np.arange(n_chans):
        c1 = c[:, k]  # correlation of channel k with all other channels
        # sort by correlation, descending order
        idx = np.argsort(c1 ** 2, 0)[::-1]
        idx = idx[skip + 1:skip + n_neighbors + 1]  # keep best

        # pca neighbors to orthogonalize them
        c2 = wc[idx, :][:, idx]
        [eigvec, eigval] = pca(c2)

        # Some of the eigenvalues could be zero or inf
        norm = np.zeros(len(eigval))
        mask = np.logical_and(eigval > eps, np.isfinite(eigval))
        norm[mask] = 1. / np.sqrt(eigval[mask])
        eigvec *= norm
        del eigval, norm, mask

        r[k, idx] = np.dot(eigvec, np.dot(wc[k][idx], eigvec))

        if r[k, k] != 0:
            raise RuntimeError('SNS operator should be zero along diagonal')

    return r.T


def sns1(X, n_neighbors=None, skip=0):
    """Sensor Noise Suppresion 1.

    This version of SNS first regresses out major shared components.
    """
    if X.ndim > 2:
        raise Exception("SNS1 works only with 2D matrices")

    n_samples, n_chans, n_trials = theshapeof(X)

    if n_neighbors is None:
        n_neighbors = n_chans - skip - 1
    else:
        n_neighbors = np.min((n_neighbors, n_chans - skip - 1))

    mn = np.mean(X)
    X = (X - mn)  # remove mean
    N = np.sqrt(np.sum(X ** 2))
    NN = 1 / N
    NN[np.where(np.isnan(NN))] = 0
    X = (X * NN)  # normalize

    y = np.zeros(X.shape)

    for k in np.arange(n_chans):
        c1 = X.T * X[:, k]  # correlation with neighbors
        c1 = c1 / c1[k]
        c1[k] = 0  # demote self
        [c1, idx] = np.sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1 + skip:n_neighbors + skip]   # keep best

        # pca neighbors to orthogonalize them
        xx = X[:, idx]
        c2 = xx.T * xx
        [eigvec, eigval] = pca(c2)
        eigvec = eigvec * np.diag(1 / np.sqrt(eigval))

        y[:, k] = tsr(X[:, k], xx * eigvec)

    y = (y * N)

    return y
