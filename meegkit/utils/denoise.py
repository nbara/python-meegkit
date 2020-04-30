"""Denoising utilities."""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec

from .matrix import fold, theshapeof, unfold, _check_weights


def demean(X, weights=None, return_mean=False):
    """Remove weighted mean over rows (samples).

    Parameters
    ----------
    X : array, shape=(n_samples, n_channels[, n_trials])
        Data.
    weights : array, shape=(n_samples)

    Returns
    -------
    demeaned_X : array, shape=(n_samples, n_channels[, n_trials])
        Centered data.
    mn : array
        Mean value.

    """
    weights = _check_weights(weights, X)
    ndims = X.ndim
    n_samples, n_chans, n_trials = theshapeof(X)
    X = unfold(X)

    if weights.any():
        weights = unfold(weights)

        if weights.shape[0] != X.shape[0]:
            raise ValueError('X and weights arrays should have same ' +
                             'number of samples (rows).')

        if weights.shape[1] == 1 or weights.shape[1] == n_chans:
            mn = (np.sum(X * weights, axis=0) /
                  np.sum(weights, axis=0))[None, :]
        else:
            raise ValueError('Weight array should have either the same ' +
                             'number of columns as X array, or 1 column.')

        demeaned_X = X - mn
    else:
        mn = np.mean(X, axis=0, keepdims=True)
        demeaned_X = X - mn

    if n_trials > 1 or ndims == 3:
        demeaned_X = fold(demeaned_X, n_samples)

    if return_mean:
        return demeaned_X, mn  # the_mean.shape=(1, the_mean.shape[0])
    else:
        return demeaned_X


def mean_over_trials(X, weights=None):
    """Compute weighted mean over trials."""
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

    if weights.size > 0:
        X = X * weights
        y = np.sum(X ** 2)
        tweight = np.sum(weights)
    else:
        y = np.sum(X ** 2)
        tweight = X.size

    return y, tweight


def find_outlier_samples(X, toobig1, toobig2=[]):
    """Find outlier trials using an absolute threshold."""
    n_samples, n_chans, n_trials = theshapeof(X)
    X = unfold(X)

    # apply absolute threshold
    weights = np.ones((n_trials * n_samples, n_chans))

    if toobig1 is not None:
        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)

        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)

        weights[np.where(abs(X) > toobig1)] = 0
        X = demean(X, weights)
    else:
        weights = np.ones(X.shape)

    # apply relative threshold
    if toobig2:
        _, mn = demean(X ** 2, weights, return_mean=True)
        X = np.tile(mn, (X.shape[0], 1))
        idx = np.where(X ** 2 > (X * toobig2))[0]
        weights[idx] = 0

    weights = fold(weights, n_samples)

    return weights


def find_outlier_trials(X, thresh=None, show=True):
    """Find outlier trials.

    For example thresh=2 rejects trials that deviate from the mean by
    more than twice the average deviation from the mean.

    Parameters
    ----------
    X : ndarray, shape=(n_times, n_chans[, n_trials])
        Data array.
    thresh : float or array of floats
        Keep trials less than thresh from mean.
    show : bool
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

    if show:
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

    thresh.pop(0)
    if thresh:
        bads2, _ = find_outlier_trials(X[:, idx], thresh, show)
        idx2 = idx[bads2]
        idx = np.setdiff1d(idx, idx2)

    bads = []
    if len(idx) < n_trials:
        bads = np.setdiff1d(range(n_trials), idx)

    return bads, d
