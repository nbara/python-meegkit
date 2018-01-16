import numpy as np

from .utils import demean, fold, pca, theshapeof, tscov, tsregress, unfold


def sns(data, n_neighbors=0, skip=0, weights=np.array([])):
    """Sensor Noise Suppresion.

    This algorithm will replace the data from each channel by its regression on
    the subspace formed by the other channels. The underlying assumptions are
    that (a) sensor noise is uncorrelated across sensors, and (b) genuine
    signal is correlated, sensor noise is removed and genuine signal is
    preserved.

    Parameters
    ----------
    data : array, shape = (n_samples, n_chans, n_trials)
        EEG data.
    n_neighbors : int
        Number of neighbors (based on correlation) to include in the
        projection.
    skip: int
        Number of closest neighbors to skip (default: 0).
    weights : array of floats
        Weights (default: all ones).


    Returns
    -------
    y : array, shape = (n_samples, n_chans, n_trials)
        Denoised data.
    r : array, shape = (n_chans, n_chans)
        Denoising matrix.

    """
    if not n_neighbors:
        n_neighbors = data.shape[1] - 1

    n_samples, n_chans, n_trials = theshapeof(data)
    data = unfold(data)

    data, _ = demean(data)
    c, nc = tscov(data)

    if weights:
        weights = unfold(weights)
        data, _ = demean(data, weights)
        wc, nwc = tscov(data, shifts=None, weights=weights)
        r = sns0(c, n_neighbors, skip, wc)
    else:
        weights = np.ones((n_chans, n_trials))
        r = sns0(c, n_neighbors, skip, c)

    y = np.dot(np.squeeze(data), r)
    y = fold(y, n_samples)

    return y, r


def sns0(c, n_neighbors=0, skip=0, wc=[]):
    """Sensor Noise Suppresion from data covariance.

    Parameters
    ----------
    c: array, shape = (n_chans, n_chans)
        Full covariance of data to denoise.
    n_neighbors : int
        Number of neighbors (based on correlation) to include in the
        projection.
    skip: int
        Number of closest neighbors to skip (default: 0).
    wc: array
        Weighted covariance.

    Returns
    -------
    r: arraym
        Denoising matrix.

    """
    if not wc.any():
        wc = c

    n_chans = c.shape[0]

    if n_neighbors == 0:
        n_neighbors = n_chans - skip - 1
    else:
        n_neighbors = np.min((n_neighbors, n_chans - skip - 1))

    r = np.zeros(c.shape)

    # normalize
    eps = np.finfo(np.float64).eps
    d = np.sqrt(1. / (np.diag(c) + eps))
    c = c * d * d.T

    for k in np.arange(n_chans):
        c1 = c[:, k]  # correlation of channel k with all other channels
        # sort by correlation, descending order
        idx = np.argsort(c1 ** 2, 0)[::-1]
        c1 = c1[idx]
        idx = idx[skip + 1:skip + n_neighbors + 1]  # keep best

        # pca neighbors to orthogonalize them
        c2 = wc[idx, :][:, idx]
        [eigvec, eigval] = pca(c2)

        # Some of the eigenvalues could be zero or inf
        norm = np.zeros(len(eigval))
        use_mask = np.logical_and(eigval > 100 * eps, np.isfinite(eigval))
        norm[use_mask] = 1. / np.sqrt(eigval[use_mask])
        eigvec *= norm
        del eigval

        r[k, idx] = np.dot(eigvec, np.dot(wc[k][idx], eigvec))

        # Alternative implementation (from noisetools)
        # augment rotation matrix to include this channel
        # stack1 = np.hstack((1, np.zeros(eigvec.shape[0])))
        # stack2 = np.hstack((np.zeros((eigvec.shape[0], 1)), eigvec))
        # eigvec = np.vstack((stack1, stack2))
        #
        # # correlation matrix for rotated data
        # c3 = np.dot(np.dot(
        #     eigvec.T, wc[np.hstack((k, idx)), :][:, np.hstack((k, idx))]),
        #     eigvec)
        #
        # # first row defines projection to clean component k
        # c4 = np.dot(c3[0, 1:], eigvec[1:, 1:].T)
        # c4.shape = (c4.shape[0], 1)
        # insert new column into denoising matrix
        # r[idx, k] = np.squeeze(c4)

        if r[k, k] != 0:
            raise RuntimeError('SNS operator should be zero along diagonal')

    return r


def sns1(x, n_neighbors=0, skip=0):
    """Sensor Noise Suppresion 1."""
    if x.ndim > 2:
        raise Exception("SNS1 works only with 2D matrices")

    n_samples, n_chans, n_trials = theshapeof(x)

    if n_neighbors == 0:
        n_neighbors = n_chans - skip - 1
    else:
        n_neighbors = np.min((n_neighbors, n_chans - skip - 1))

    mn = np.mean(x)
    x = (x - mn)  # remove mean
    N = np.sqrt(np.sum(x ** 2))
    NN = 1 / N
    NN[np.where(np.isnan(NN))] = 0
    x = (x * NN)  # normalize

    y = np.zeros(x.shape)

    for k in np.arange(n_chans):

        c1 = x.T * x[:, k]  # correlation with neighbors
        c1 = c1 / c1[k]
        c1[k] = 0                           # demote self
        [c1, idx] = np.sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1 + skip:n_neighbors + skip]   # keep best

        # pca neighbors to orthogonalize them
        xx = x[:, idx]
        c2 = xx.T * xx
        [eigvec, eigval] = pca(c2)
        eigvec = eigvec * np.diag(1 / np.sqrt(eigval))

        y[:, k] = tsregress(x[:, k], xx * eigvec)

    y = (y * N)

    return y
