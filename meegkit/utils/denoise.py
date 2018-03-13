from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import linalg

from .matrix import fold, theshapeof, unfold, demean
from .covariances import tscov, tsxcov


def pca(cov, max_comps=None, thresh=0):
    """PCA from covariance.

    Parameters
    ----------
    cov:  array, shape = (n_chans, n_chans)
        Covariance matrix.
    max_comps : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float
        Discard components below this threshold.

    Returns
    -------
    V : array, shape = (max_comps, max_comps)
        Eigenvectors (matrix of PCA components).
    d : array, shape = (max_comps,)
        PCA eigenvalues

    """
    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')

    d, V = linalg.eigh(cov)
    d = d.real
    V = V.real

    idx = np.argsort(d)[::-1]  # reverse sort ev order
    d = d[idx]
    V = V[:, idx]

    # Truncate weak components
    if thresh is not None:
        idx = np.where(d / d.max() > thresh)[0]
        d = d[idx]
        V = V[:, idx]

    # Keep a fixed number of components
    if max_comps is None:
        max_comps = V.shape[1]
    else:
        max_comps = np.min(max_comps, V.shape[1])

    V = V[:, np.arange(max_comps)]
    d = d[np.arange(max_comps)]

    return V, d


def regcov(cxy, cyy, keep=np.array([]), threshold=np.array([])):
    """Compute regression matrix from cross covariance."""
    # PCA of regressor
    [V, d] = pca(cyy)

    # discard negligible regressor PCs
    if keep:
        keep = max(keep, V.shape[1])
        V = V[:, 0:keep]
        d = d[0:keep]

    if threshold:
        idx = np.where(d / max(d) > threshold)
        V = V[:, idx]
        d = d[idx]

    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = np.dot(V.T, cxy)

    # projection matrix from regressor PCs
    r = (r.T * 1 / d).T

    # projection matrix from regressors
    r = np.dot(np.squeeze(V), np.squeeze(r))

    return r


def tsregress(X, y, shifts=np.array([0]), keep=np.array([]),
              threshold=np.array([]), toobig1=np.array([]),
              toobig2=np.array([])):
    """Time-shift regression."""
    # shifts must be non-negative
    mn = shifts.min()
    if mn < 0:
        shifts = shifts - mn
        X = X[-mn + 1:, :, :]
        y = y[-mn + 1:, :, :]

    n_shifts = shifts.size

    # flag outliers in X and y
    if toobig1 or toobig2:
        xw = find_outliers(X, toobig1, toobig2)
        yw = find_outliers(y, toobig1, toobig2)
    else:
        xw = []
        yw = []

    if X.ndim == 3:
        [Mx, Nx, Ox] = X.shape
        [My, Ny, Oy] = y.shape
        X = unfold(X)
        y = unfold(y)
        [X, xmn] = demean(X, xw)
        [y, ymn] = demean(y, yw)
        X = fold(X, Mx)
        y = fold(y, My)
    else:
        [X, xmn] = demean(X, xw)
        [y, ymn] = demean(y, yw)

    # covariance of y
    [cyy, totalweight] = tscov(y, shifts.T, yw)
    cyy = cyy / totalweight

    # cross-covariance of X and y
    [cxy, totalweight] = tsxcov(X, y, shifts.T, xw, yw)
    cxy = cxy / totalweight

    # regression matrix
    r = regcov(cxy, cyy, keep, threshold)

    # regression
    if X.ndim == 3:
        X = unfold(X)
        y = unfold(y)

        [n_samples, n_chans, n_trials] = X.shape
        mm = n_samples - max(shifts)
        z = np.zeros(X.shape)

        for k in range(n_shifts):
            kk = shifts(k)
            idx1 = np.r_[kk + 1:kk + mm]
            idx2 = k + np.r_[0:y.shape[1]] * n_shifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]

        z = fold(z, Mx)
        z = z[0:-max(shifts), :, :]
    else:
        n_samples, n_chans = X.shape
        z = np.zeros((n_samples - max(shifts), n_chans))
        for k in range(n_shifts):
            kk = shifts(k)
            idx1 = np.r_[kk + 1:kk + z.shape[0]]
            idx2 = k + np.r_[0:y.shape[1]] * n_shifts
            z = z + y[idx1, :] * r[idx2, :]

    offset = max(0, -mn)
    idx = np.r_[offset + 1:offset + z.shape[0]]

    return z, idx


def wmean(X, weights=[], axis=0):
    """Weighted mean."""
    if not weights:
        y = np.mean(X, axis)
    else:
        if X.shape[0] != weights.shape[0]:
            raise Exception("data and weight must have same nrows")
        if weights.shape[1] == 1:
            weights = np.tile(weights, (1, X.shape(1)))
        if weights.shape[1] != X.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")

        y = np.sum(X * weights, axis) / np.sum(weights, axis)

    return y


def mean_over_trials(X, weights=None):
    """Compute mean over trials."""
    if weights is None:
        weights = np.array([])

    n_samples, n_chans, n_trials = theshapeof(X)

    if not weights.any():
        y = np.mean(X, 2)
        tw = np.ones((n_samples, n_chans, 1)) * n_trials
    else:
        m, n, o = theshapeof(weights)
        if m != n_samples:
            raise "!"
        if o != n_trials:
            raise "!"

        X = unfold(X)
        weights = unfold(weights)

        if n == n_chans:
            X = X * weights
            X = fold(X, n_samples)
            weights = fold(weights, n_samples)
            y = np.sum(X, 3) / np.sum(weights, 3)
        elif n == 1:
            X = X * weights
            X = fold(X, n_samples)
            weights = fold(weights, n_samples)
            y = np.sum(X, 3) * 1 / np.sum(weights, 3)

        tw = np.sum(weights, 3)

    return y, tw


def wpwr(X, weights=None):
    """Weighted power."""
    if weights is None:
        weights = np.array([])

    X = unfold(X)
    weights = unfold(weights)

    if weights:
        X = X * weights
        y = np.sum(X ** 2)
        tweight = np.sum(weights)
    else:
        y = np.sum(X ** 2)
        tweight = X.size

    return y, tweight


def find_outliers(X, toobig1, toobig2=[]):
    """Find outlier trials using an absolute threshold."""
    n_samples, n_chans, n_trials = theshapeof(X)

    # remove mean
    X = unfold(X)
    X = demean(X)[0]

    # apply absolute threshold
    weights = np.ones(X.shape)
    if toobig1:
        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)[0]

        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)[0]

        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)[0]
    else:
        weights = np.ones(X.shape)

    # apply relative threshold
    if toobig2:
        X = wmean(X ** 2, weights)
        X = np.tile(X, (X.shape[0], 1))
        idx = np.where(X**2 > (X * toobig2))
        weights[idx] = 0

    weights = fold(weights, n_samples)

    return weights


def find_outlier_trials(X, thresh=None, disp_flag=True):
    """Find outlier trials.

    For example thresh=2 rejects trials that deviate from the mean by
    more than twice the average deviation from the mean.

    Parameters
    ----------
    X : ndarray, shape = (n_times, n_chans[, n_trials])
        Data array.
    thresh : float or array of floats
        Keep trials less than thresh from mean.
    disp_flag : bool
        If true plot trial deviations before and after.

    Returns
    -------
    bads : list of int
        Indices of trials to reject.
    d : array
        Relative deviations from mean.

    """
    if thresh is None:
        thresh = [np.inf]
    elif isinstance(thresh, float) or isinstance(thresh, int):
        thresh = [thresh]

    if X.ndim > 3:
        raise ValueError('X should be 2D or 3D')
    elif X.ndim == 3:
        n_samples, n_chans, n_trials = theshapeof(X)
        X = np.reshape(X, (n_samples * n_chans, n_trials))
    else:
        n_chans, n_trials = X.shape

    avg = np.mean(X, axis=-1, keepdims=True)  # mean over trials
    d = X - avg  # difference from mean
    d = np.sum(d ** 2, axis=0)

    d = d / (np.sum(X ** 2) / n_trials)
    idx = np.where(d < thresh[0])[0]

    if disp_flag:
        plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1, 2)
        plt.suptitle('Outlier trial detection')

        # Before
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(d, ls='-')
        ax1.plot(np.setdiff1d(np.arange(n_trials), idx),
                 d[np.setdiff1d(np.arange(n_trials), idx)], color='r', ls=' ',
                 marker='.')
        ax1.axhline(y=thresh[0], color='grey', linestyle=':')
        ax1.set_xlabel('Trial #')
        ax1.set_ylabel('Normalized deviation from mean')
        ax1.set_title('Before, ' + str(len(d)), fontsize=10)
        ax1.set_xlim(0, len(d) + 1)
        plt.draw()

        # After
        ax2 = plt.subplot(gs[0, 1])
        _, dd = find_outlier_trials(X[:, idx], None, False)
        ax2.plot(dd, ls='-')
        ax2.set_xlabel('Trial #')
        ax2.set_title('After, ' + str(len(idx)), fontsize=10)
        ax2.yaxis.tick_right()
        ax2.set_xlim(0, len(idx) + 1)
        plt.show()

    thresh.pop()
    if thresh:
        bads2, _ = find_outlier_trials(X[:, idx], thresh, disp_flag)
        idx2 = idx[bads2]
        idx = np.setdiff1d(idx, idx2)

    bads = []
    if len(idx) < n_trials:
        bads = np.setdiff1d(range(n_trials), idx)

    return bads, d
