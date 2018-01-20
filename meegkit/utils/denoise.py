from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import linalg

from .matrix import fold, theshapeof, unfold, demean
from .covariances import tscov, tsxcov


def pca(cov, max_components=None, thresh=0):
    """PCA rotation from covariance.

    Parameters
    ----------
    cov:  array, shape = (n_chans, n_chans)
        Covariance matrix.
    max_components : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).

    Returns
    -------
    eigenvectors: array, shape = (max_components, max_components)
        Eigenvectors (matrix of PCA components).
    eigenvalues: PCA eigenvalues

    """
    if not max_components:
        max_components = cov.shape[0]  # keep all components
    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')

    eigenvalues, eigenvector = linalg.eig(cov)

    eigenvalues = eigenvalues.real
    eigenvector = eigenvector.real

    idx = np.argsort(eigenvalues)[::-1]  # reverse sort ev order
    eigenvalues = eigenvalues[idx]

    # Truncate
    eigenvectors = eigenvector[:, idx]
    eigenvectors = eigenvectors[:, np.arange(max_components)]
    eigenvalues = eigenvalues[np.arange(max_components)]

    if thresh is not None:
        suprathresh = np.where(eigenvalues / eigenvalues.max() > thresh)[0]
        eigenvalues = eigenvalues[suprathresh]
        eigenvectors = eigenvectors[:, suprathresh]

    return eigenvectors, eigenvalues


def regcov(cxy, cyy, keep=np.array([]), threshold=np.array([])):
    """Compute regression matrix from cross covariance."""
    # PCA of regressor
    [topcs, eigenvalues] = pca(cyy)

    # discard negligible regressor PCs
    if keep:
        keep = max(keep, topcs.shape[1])
        topcs = topcs[:, 0:keep]
        eigenvalues = eigenvalues[0:keep]

    if threshold:
        idx = np.where(eigenvalues / max(eigenvalues) > threshold)
        topcs = topcs[:, idx]
        eigenvalues = eigenvalues[idx]

    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = np.dot(topcs.T, cxy)

    # projection matrix from regressor PCs
    r = (r.T * 1 / eigenvalues).T

    # projection matrix from regressors
    r = np.dot(np.squeeze(topcs), np.squeeze(r))

    return r


def tsregress(x, y, shifts=np.array([0]), keep=np.array([]),
              threshold=np.array([]), toobig1=np.array([]),
              toobig2=np.array([])):
    """Time-shift regression."""
    # shifts must be non-negative
    mn = shifts.min()
    if mn < 0:
        shifts = shifts - mn
        x = x[-mn + 1:, :, :]
        y = y[-mn + 1:, :, :]

    n_shifts = shifts.size

    # flag outliers in x and y
    if toobig1 or toobig2:
        xw = find_outliers(x, toobig1, toobig2)
        yw = find_outliers(y, toobig1, toobig2)
    else:
        xw = []
        yw = []

    if x.ndim == 3:
        [Mx, Nx, Ox] = x.shape
        [My, Ny, Oy] = y.shape
        x = unfold(x)
        y = unfold(y)
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)
        x = fold(x, Mx)
        y = fold(y, My)
    else:
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)

    # covariance of y
    [cyy, totalweight] = tscov(y, shifts.T, yw)
    cyy = cyy / totalweight

    # cross-covariance of x and y
    [cxy, totalweight] = tsxcov(x, y, shifts.T, xw, yw)
    cxy = cxy / totalweight

    # regression matrix
    r = regcov(cxy, cyy, keep, threshold)

    # regression
    if x.ndim == 3:
        x = unfold(x)
        y = unfold(y)

        [n_samples, n_chans, n_trials] = x.shape
        mm = n_samples - max(shifts)
        z = np.zeros(x.shape)

        for k in range(n_shifts):
            kk = shifts(k)
            idx1 = np.r_[kk + 1:kk + mm]
            idx2 = k + np.r_[0:y.shape[1]] * n_shifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]

        z = fold(z, Mx)
        z = z[0:-max(shifts), :, :]
    else:
        n_samples, n_chans = x.shape
        z = np.zeros((n_samples - max(shifts), n_chans))
        for k in range(n_shifts):
            kk = shifts(k)
            idx1 = np.r_[kk + 1:kk + z.shape[0]]
            idx2 = k + np.r_[0:y.shape[1]] * n_shifts
            z = z + y[idx1, :] * r[idx2, :]

    offset = max(0, -mn)
    idx = np.r_[offset + 1:offset + z.shape[0]]

    return z, idx


def wmean(x, weights=[], axis=0):
    """Weighted mean."""
    if not weights:
        y = np.mean(x, axis)
    else:
        if x.shape[0] != weights.shape[0]:
            raise Exception("data and weight must have same nrows")
        if weights.shape[1] == 1:
            weights = np.tile(weights, (1, x.shape(1)))
        if weights.shape[1] != x.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")

        y = np.sum(x * weights, axis) / np.sum(weights, axis)

    return y


def mean_over_trials(x, weights=None):
    """Compute mean over trials."""
    if weights is None:
        weights = np.array([])

    n_samples, n_chans, n_trials = theshapeof(x)

    if not weights.any():
        y = np.mean(x, 2)
        tw = np.ones((n_samples, n_chans, 1)) * n_trials
    else:
        m, n, o = theshapeof(weights)
        if m != n_samples:
            raise "!"
        if o != n_trials:
            raise "!"

        x = unfold(x)
        weights = unfold(weights)

        if n == n_chans:
            x = x * weights
            x = fold(x, n_samples)
            weights = fold(weights, n_samples)
            y = np.sum(x, 3) / np.sum(weights, 3)
        elif n == 1:
            x = x * weights
            x = fold(x, n_samples)
            weights = fold(weights, n_samples)
            y = np.sum(x, 3) * 1 / np.sum(weights, 3)

        tw = np.sum(weights, 3)

    return y, tw


def wpwr(x, weights=None):
    """Weighted power."""
    if weights is None:
        weights = np.array([])

    x = unfold(x)
    weights = unfold(weights)

    if weights:
        x = x * weights
        y = np.sum(x ** 2)
        tweight = np.sum(weights)
    else:
        y = np.sum(x ** 2)
        tweight = x.size

    return y, tweight


def find_outliers(x, toobig1, toobig2=[]):
    """Find outlier trials using an absolute threshold."""
    n_samples, n_chans, n_trials = theshapeof(x)

    # remove mean
    x = unfold(x)
    x = demean(x)[0]

    # apply absolute threshold
    weights = np.ones(x.shape)
    if toobig1:
        weights[np.where(abs(x) > toobig1)] = 0
        x = demean(x, weights)[0]

        weights[np.where(abs(x) > toobig1)] = 0
        x = demean(x, weights)[0]

        weights[np.where(abs(x) > toobig1)] = 0
        x = demean(x, weights)[0]
    else:
        weights = np.ones(x.shape)

    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, weights)
        X = np.tile(X, (x.shape[0], 1))
        idx = np.where(x**2 > (X * toobig2))
        weights[idx] = 0

    weights = fold(weights, n_samples)

    return weights


def find_outlier_trials(x, thresh=None, disp_flag=True):
    """Find outlier trials.

    For example thresh=2 rejects trials that deviate from the mean by
    more than twice the average deviation from the mean.

    Parameters
    ----------
    x : ndarray
        Data array (n_trials * n_chans * time).
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

    if x.ndim > 3:
        raise ValueError('x should be 2D or 3D')
    elif x.ndim == 3:
        n_chans, n_chans, n_trials = x.shape  # n_trials * n_chans * time
        x = np.reshape(x, (n_chans, n_chans * n_trials))
    else:
        n_chans, _ = x.shape

    n_samples = np.mean(x, axis=0)  # mean over trials
    n_samples = np.tile(n_samples, (n_chans, 1))  # repeat mean
    d = x - n_samples  # difference from mean
    dd = np.zeros(n_chans)
    for i_trial in range(n_chans):
        dd[i_trial] = np.sum(d[i_trial, :] ** 2)
    d = dd / (np.sum(x.flatten() ** 2) / n_chans)
    idx = np.where(d < thresh[0])[0]
    del dd

    if disp_flag:
        plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1, 2)

        plt.suptitle('Outlier trial detection')

        # Before
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(d, ls='-')
        ax1.plot(np.setdiff1d(np.arange(n_chans), idx),
                 d[np.setdiff1d(np.arange(n_chans), idx)], color='r', ls=' ',
                 marker='.')
        ax1.axhline(y=thresh[0], color='grey', linestyle=':')
        ax1.set_xlabel('Trial #')
        ax1.set_ylabel('Normalized deviation from mean')
        ax1.set_title('Before, ' + str(len(d)), fontsize=10)
        ax1.set_xlim(0, len(d) + 1)
        plt.draw()

        # After
        ax2 = plt.subplot(gs[0, 1])
        _, dd = find_outlier_trials(x[idx, :], None, False)
        ax2.plot(dd, ls='-')
        ax2.set_xlabel('Trial #')
        ax2.set_title('After, ' + str(len(idx)), fontsize=10)
        ax2.yaxis.tick_right()
        ax2.set_xlim(0, len(idx) + 1)
        plt.show()

    thresh.pop()
    if thresh:
        bads2, _ = find_outlier_trials(x[idx, :], thresh, disp_flag)
        idx2 = idx[bads2]
        idx = np.setdiff1d(idx, idx2)

    bads = []
    if len(idx) < n_chans:
        bads = np.setdiff1d(range(n_chans), idx)

    return bads, d
