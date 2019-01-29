"""Robust detrending."""
import numpy as np

from .utils import demean, pca, unfold
from .utils.matrix import _check_weights


def regress(y, x, w=None, threshold=1e-7, return_mean=False):
    """Weighted regression.

    Parameters
    ----------
    y : array, shape=(n_times, n_chans)
        Data.
    x : array, shape=(n_times, n_chans)
        Regressor.
    w :
        Weight to apply to `y`. `w` is either a matrix of same size as `y`, or
        a column vector to be applied to each column of `y`.
    threshold : float
        PCA threshold (default=1e-7).
    return_mean : bool
        If True, also return the signal mean prior to regression.

    Returns
    -------
    b : array, shape=(n_chans, n_chans)
        Regression matrix (apply to x to approximate y).
    z : array, shape=(n_times, n_chans)
        Regression (x @ b).

    """
    # check/fix sizes
    w = _check_weights(w, y)
    n_times = y.shape[0]
    n_chans = y.shape[1]
    x = unfold(x)
    y = unfold(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y have incompatible shapes!')

    # save weighted mean
    mn = y - demean(y, w)

    if not w.any():  # simple regression
        xx = demean(x)
        yy = demean(y)

        # PCA
        V, _ = pca(xx.T.dot(xx), thresh=threshold)
        xxx = xx.dot(V)
        b = yy.T.dot(xxx) / xxx.T.dot(xxx)
        b = b.T
        z = np.dot(demean(x, w).dot(V), b)
        z = z + mn

    else:  # weighted regression
        if w.shape[0] != n_times:
            raise ValueError('!')

        if w.shape[1] == 1:  # same weight for all channels
            if sum(w.flatten()) == 0:
                print('weights all zero')
                b = 0
            else:
                yy = demean(y, w) * w
                xx = demean(x, w) * w
                V, _ = pca(xx.T.dot(xx), thresh=threshold)
                xxx = xx.dot(V)
                b = yy.T.dot(xxx) / xxx.T.dot(xxx)

            z = demean(x, w).dot(V).dot(b.T)
            z = z + mn

        else:  # each channel has own weight
            if w.shape[1] != y.shape[1]:
                raise ValueError('!')
            z = np.zeros(y.shape)
            b = np.zeros((n_chans, n_chans))
            for i in range(n_chans):
                if sum(w[:, i]) == 0:
                    print('weights all zero for channel {}'.format(i))
                    c = np.zeros(y.shape[1], 1)
                else:
                    wc = w[:, i][:, None]  # channel-specific weight
                    yy = demean(y[:, i], wc) * wc
                    # remove channel-specific-weighted mean from regressor
                    x = demean(x, wc)
                    xx = x * wc
                    V, _ = pca(xx.T.dot(xx), thresh=threshold)
                    xx = xx.dot(V)
                    c = yy.T.dot(xx) / xx.T.dot(xx)

                z[:, i] = x.dot(V.dot(c.T)).flatten()
                z[:, i] += mn[:, i]
                b[i] = c
            b = b[:, :V.shape[1]]

    return b, z
