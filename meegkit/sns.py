import numpy as np
import scipy.linalg

from .utils import demean, fold, pcarot, theshapeof, tscov, tsregress, unfold


def sns(data, nneighbors=0, skip=0, w=np.array([])):
    """docstring for sns"""
    if not nneighbors:
        nneighbors = data.shape[1] - 1

    m, n, o = data.shape
    data = unfold(data)

    data, mn0 = demean(data)
    c, nc = tscov(data)

    if w:
        w = unfold(w)
        data, mn1 = demean(data, w)
        wc, nwc = tscov(data, [], w)
        r = sns0(c, nneighbors, skip, wc)
    else:
        mn1 = 0
        w = np.ones((n, o))
        r = sns0(c, nneighbors, skip, c)

    y = np.dot(np.squeeze(data), r)
    y = fold(y, m)

    mn = mn0 + mn1

    return y


def sns0(c, nneighbors, skip=0, wc=[]):
    """Sensor Noise Suppresion."""

    if not any(wc):
        wc = c

    n = c.shape[0]

    if not nneighbors:
        nneighbors = n - 1

    r = np.zeros(c.shape)

    # normalize
    d = np.sqrt(1 / np.diag(c))
    c = c * d * d.T

    for k in np.xrange(n):
        c1 = c[:, k]  # correlation of channel k with all other channels
        # sort by correlation, descending order
        idx = np.argsort(c1**2, 0)[::-1]
        c1 = c1[idx]
        idx = idx[skip:skip + nneighbors]  # keep best
        # print "c1", c1.shape
        # print "idx", idx

        # pca neighbors to orthogonalize them
        c2 = wc[idx, :][:, idx]
        [topcs, eigenvalues] = pcarot(c2)
        topcs = np.dot(topcs, np.diag(1 / np.sqrt(eigenvalues)))
        # print "c2", c2.shape

        # augment rotation matrix to include this channel
        stack1 = np.hstack((1, np.zeros(topcs.shape[0])))
        stack2 = np.hstack((np.zeros((topcs.shape[0], 1)), topcs))
        topcs = np.vstack((stack1, stack2))

        # correlation matrix for rotated data
        # c3 = topcs.T * wc[hstack((k,idx)), hstack((k,idx))] * topcs
        c3 = np.dot(np.dot(
            topcs.T, wc[np.hstack((k, idx)), :][:, np.hstack((k, idx))]),
            topcs)
        # print "c3", c3.shape

        # first row defines projection to clean component k
        c4 = np.dot(c3[0, 1:], topcs[1:, 1:].T)
        c4.shape = (c4.shape[0], 1)
        # print "c4", c4

        # insert new column into denoising matrix
        r[idx, k] = np.squeeze(c4)

    return r


def sns1(x, nneighbors, skip):
    """docstring for sns1"""
    if x.ndim > 2:
        raise Exception("SNS1 works only with 2D matrices")

    m, n, o = theshapeof(x)

    if not nneighbors:
        nneighbors = n - 1

    if not skip:
        skip = 0

    mn = np.mean(x)
    x = (x - mn)  # remove mean
    N = np.sqrt(np.sum(x ** 2))
    NN = 1 / N
    NN[np.where(np.isnan(NN))] = 0
    x = (x * NN)  # normalize

    y = np.zeros(x.shape)

    for k in np.xrange(n):

        c1 = x.T * x[:, k]  # correlation with neighbors
        c1 = c1 / c1[k]
        c1[k] = 0                           # demote self
        [c1, idx] = np.sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1 + skip:nneighbors + skip]   # keep best

        # pca neighbors to orthogonalize them
        xx = x[:, idx]
        c2 = xx.T * xx
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * np.diag(1 / np.sqrt(eigenvalues))

        y[:, k] = tsregress(x[:, k], xx * topcs)

        # if mod(k,1000) == 0:
        # [k 100 * sum(y[:,0:k] ** 2) / sum(x[:, 0:k] ** 2)]

    y = (y * N)

    return y
