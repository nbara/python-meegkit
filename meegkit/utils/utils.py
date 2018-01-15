from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy import linalg

from .matrix import fold, theshapeof, unfold, unsqueeze


def rms(x, axis=0):
    """Root-mean-square along given axis."""
    return np.sqrt(np.mean(x ** 2, axis=axis, keepdims=True))


def multishift(data, shifts):
    """Apply multiple shifts to an array."""
    if min(shifts) > 0:
        raise Exception('shifts should be non-negative')

    data = unsqueeze(data)
    _, n_chans, n_trials = theshapeof(data)

    # shifts = shifts.T
    shifts_length = shifts.size

    # array of shift indices
    N = data.shape[0] - max(shifts)
    shiftarray = ((np.ones((N, shifts_length), int) * shifts).T + np.r_[0:N]).T
    z = np.zeros((N, n_chans * shifts_length, n_trials))

    for trial in range(n_trials):
        for channel in range(n_chans):
            y = data[:, channel, trial]
            a = channel * shifts_length
            b = channel * shifts_length + shifts_length
            z[:, np.arange(a, b), trial] = y[shiftarray]

    return z


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


def tscov(data, shifts=None, weights=None):
    """Time shift covariance.

    This function calculates, for each pair [DATA[i], DATA[j]] of columns of
    DATA, the cross-covariance matrix between the time-shifted versions of
    DATA[i]. Shifts are taken from array SHIFTS. Weights are taken from
    `weights`.

    DATA can be 1D, 2D or 3D.  WEIGHTS is 1D (if DATA is 1D or 2D) or
    2D (if DATA is 3D).

    Output is a 2D matrix with dimensions (ncols(X)*n_shifts)^2.
    This matrix is made up of a DATA.shape[1]^2 matrix of submatrices
    of dimensions n_shifts**2.

    The weights are not shifted.

    Parameters
    ----------
    data: array
        Data.
    shifts: array
        Array of time shifts (must be non-negative).
    weights: array
        Weights.

    Returns
    -------
    covariance: array
        Covariance matrix.
    total_weight: array
        Total weight (covariance/total_weight is normalized covariance).

    """
    if shifts is None:
        shifts = np.array([0])
    if weights is None:
        weights = np.array([])
    if shifts.min() < 0:
        raise ValueError("Shifts should be non-negative.")

    n_shifts = np.size(shifts)

    n_samples, n_chans, n_trials = theshapeof(data)
    data = unsqueeze(data)
    covariance = np.zeros((n_chans * n_shifts, n_chans * n_shifts))

    if weights.any():  # weights
        if weights.shape[1] > 1:
            raise ValueError("Weights array should have a single column.")

        weights = unsqueeze(weights)
        print(data.shape)
        for trial in range(n_trials):
            shifted_trial = multishift(data[..., trial], shifts)
            shifted_weight = multishift(weights[..., trial], shifts)
            shifted_trial = (np.squeeze(shifted_trial).T *
                             np.squeeze(shifted_weight)).T
            covariance += np.dot(shifted_trial.T, shifted_trial)

        total_weight = np.sum(weights[:])
    else:  # no weights
        for trial in range(n_trials):
            if data.ndim == 3:
                shifted_trial = np.squeeze(
                    multishift(data[:, :, trial], shifts))
            else:
                shifted_trial = multishift(data[:, trial], shifts)

            covariance += np.dot(shifted_trial.T, shifted_trial)

        total_weight = shifted_trial.shape[0] * n_trials

    return covariance, total_weight


def demean(data, weights=None):
    """Remove weighted mean over columns."""
    if weights is None:
        weights = np.array([])

    n_samples, n_chans, n_trials = theshapeof(data)
    data = unfold(data)

    if weights.any():
        weights = unfold(weights)

        if weights.shape[0] != data.shape[0]:
            raise ValueError('Data and weights arrays should have same ' +
                             'number of rows and pages.')

        if weights.shape[1] == 1 or weights.shape[1] == n_chans:
            the_mean = np.sum(data * weights) // np.sum(weights)
        else:
            raise ValueError('Weight array should have either the same ' +
                             'number of columns as data array, or 1 column.')

        demeaned_data = data - the_mean
    else:
        the_mean = np.mean(data, 0)
        demeaned_data = data - the_mean

    demeaned_data = fold(demeaned_data, n_samples)

    # the_mean.shape = (1, the_mean.shape[0])
    return demeaned_data, the_mean


def normcol(data, weights=None):
    """Normalize each column so its weighted msq is 1.

    If DATA is 3D, pages are concatenated vertically before calculating the
    norm.

    Weight should be either a column vector, or a matrix (2D or 3D) of same
    size as data.

    Parameters
    ----------
    data: data to normalize
    weights: weight

    Returns
    -------
    normalized_data: normalized data

    """
    if data.ndim == 3:
        n_samples, n_chans, n_trials = data.shape
        data = unfold(data)
        if not weights.any():
            # no weights
            normalized_data = fold(normcol(data), n_samples)
        else:
            if weights.shape[0] != n_samples:
                raise ValueError("Weight array should have same number of' \
                                 'columns as data array.")

            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = np.tile(weights, (1, n_samples, n_trials))

            if weights.shape != weights.shape:
                raise ValueError("Weight array should have be same shape ' \
                                 'as data array")

            weights = unfold(weights)

            normalized_data = fold(normcol(data, weights), n_samples)
    else:
        n_samples, n_chans, n_trials = theshapeof(data)
        if not weights.any():
            normalized_data = data * ((np.sum(data ** 2) / n_samples) ** -0.5)
        else:
            if weights.shape[0] != data.shape[0]:
                raise ValueError('Weight array should have same number of ' +
                                 'columns as data array.')

            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = np.tile(weights, (1, n_chans))

            if weights.shape != data.shape:
                raise ValueError('Weight array should have be same shape as ' +
                                 'data array')

            if weights.shape[1] == 1:
                weights = np.tile(weights, (1, n_chans))

            normalized_data = data * \
                (np.sum((data ** 2) * weights) / np.sum(weights)) ** -0.5

    return normalized_data


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


def tsxcov(x, y, shifts=None, weights=np.array([])):
    """Calculate cross-covariance of X and time-shifted Y.

    This function calculates, for each pair of columns (Xi,Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj.
    Shifts are taken from array SHIFTS.

    The weights are applied to X.

    X can be 1D, 2D or 3D.  W is 1D (if X is 1D or 2D) or 2D (if X is 3D).

    Output is a 2D matrix with dimensions ncols(X)*(ncols(Y)*n_shifts).

    Parameters
    ----------
    x, y: arrays
        data to cross correlate
    shifts: array
        time shifts (must be non-negative)
    weights: array
        weights

    Returns
    -------
    c: cross-covariance matrix
    tw: total weight

    """
    if shifts is None:
        shifts = np.array([0])

    n_shifts = shifts.size

    mx, nx, ox = theshapeof(x)
    my, ny, oy = theshapeof(y)
    c = np.zeros((nx, ny * n_shifts))

    if weights.any():
        x = fold(unfold(x) * unfold(weights), mx)

    # cross covariance
    for trial in range(ox):
        yy = np.squeeze(multishift(y[:, :, trial], shifts))
        xx = np.squeeze(x[0:yy.shape[0], :, trial])

        c += np.dot(xx.T, yy)

    if not weights.any():
        tw = ox * ny * yy.shape[0]
    else:
        weights = weights[0:yy.shape[0], :, :]
        tw = np.sum(weights[:])

    return c, tw


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
