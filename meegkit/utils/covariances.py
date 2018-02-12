import warnings
import numpy as np

from .matrix import (multishift, theshapeof, unsqueeze, relshift,
                     _check_shifts)


def cov_lags(X, Y, shifts=None):
    """Empirical covariance of [X,Y] with lags.

    Parameters
    ----------
    X: array, shape = (n_times, n_chans_x[, n_trials])
        Data.
    Y: array, shape = (n_times, n_chans_y[, n_trials])
        Time shifted data.
    shifts: array, shape = (n_shifts,)
        Positive lag means Y is delayed relative to X.

    Returns
    -------
    C : array, shape = (n_chans_x + n_chans_y, n_chans_x + n_chans_y, n_shifts)
        Covariance matrix (3D if n_shifts > 1).
    tw : float
        Total weight.
    m : int
        Number of columns in X.

    See Also
    --------
    relshift, tscov, tsxcov

    """
    shifts, n_shifts = _check_shifts(shifts)
    X, Y = unsqueeze(X), unsqueeze(Y)

    n_samples, n_chans, n_trials = theshapeof(X)
    n_samples2, n_chans2, n_trials2 = theshapeof(Y)

    if n_samples != n_samples2:
        raise AttributeError('X and Y must have same n_times')
    if n_trials != n_trials2:
        raise AttributeError('X and Y must have same n_trials')
    if n_samples <= max(shifts):
        raise AttributeError('shifts should be no larger than n_samples')

    n_cov = n_chans + n_chans2  # sum of channels of X and Y
    C = np.zeros((n_cov, n_cov, n_shifts))
    for t in np.arange(n_trials):
        for i, s in enumerate(shifts):
            YY, XX = relshift(Y[..., t], ref=X[..., t], shifts=s)
            XY = np.hstack((XX, YY))
            C[:, :, i] += np.dot(XY.T, XY)

    if n_shifts == 1:
        C = np.squeeze(C, 2)

    tw = n_samples * n_trials

    return C, tw, n_chans


def tsxcov(X, Y, shifts=None, weights=None):
    """Calculate cross-covariance of X and time-shifted Y.

    This function calculates, for each pair of columns (Xi, Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj.

    Output is a 2D matrix with dimensions .

    Parameters
    ----------
    X, Y : arrays, shape = (n_times, n_chans[, n_trials])
        Data to cross correlate. X can be 1D, 2D or 3D.
    shifts : array
        Time shifts.
    weights : array
        The weights that are applied to X. 1D (if X is 1D or 2D) or 2D (if X is
        3D).

    Returns
    -------
    C : array, shape = (n_chans_x, n_chans_y * n_shifts)
        Cross-covariance matrix.
    tw : total weight

    """
    weights = _check_weights(weights, X)
    shifts, n_shifts = _check_shifts(shifts)

    n_times, n_chans, n_trials = theshapeof(X)
    n_times2, n_chans2, n_trials2 = theshapeof(Y)
    X = unsqueeze(X)
    Y = unsqueeze(Y)

    # Apply weights if any
    if weights.any():
        X = np.einsum('ijk,ilk->ijk', X, weights)  # element-wise mult
        weights = weights[:n_times2, :, :]
        tw = np.sum(weights[:])
    else:  # infer weights
        N = 0
        if len(shifts[shifts < 0]):
            N -= np.min(shifts)
        if len(shifts[shifts >= 0]):
            N += np.max(shifts)
        tw = (n_chans * n_shifts - N) * n_trials

    # cross covariance
    # C = np.zeros((n_chans * n_shifts, n_chans2 * n_shifts))
    # for t in np.arange(n_trials):
    #     YY, XX = relshift(Y[..., t], ref=X[..., t], shifts=shifts)
    #     XX = XX.reshape(n_times, n_chans * n_shifts)
    #     YY = YY.reshape(n_times2, n_chans2 * n_shifts)
    #     C += np.dot(XX.T, YY)
    C = np.zeros((n_chans, n_chans2 * n_shifts))
    for t in np.arange(n_trials):
        YY = multishift(Y[..., t], shifts=shifts)
        YY = YY.reshape(n_times2, n_chans2 * n_shifts)
        C += np.dot(X[..., t].T, YY)

    return C, tw


def tscov(X, shifts=None, weights=None, assume_centered=True):
    """Time shift covariance.

    This function calculates, for each pair [X[i], X[j]] of columns of X, the
    cross-covariance matrix between the time-shifted versions of X[i].

    Parameters
    ----------
    X : array, shape = (n_times, n_chans[, n_trials])
        Data, can be 1D, 2D or 3D.
    shifts : array
        Array of time shifts.
    weights : array
        Weights, 1D (if X is 1D or 2D) or 2D (if X is 3D). The weights are not
        shifted.

    Returns
    -------
    C : array, shape = (n_channels * n_shifts, n_channels * n_shifts)
        Covariance matrix. This matrix is made up of a (n_times, n_times)
        matrix of submatrices of dimensions (n_shifts, n_shifts).
    tw : array
        Total weight (C/tw is the normalized covariance).

    """
    weights = _check_weights(weights, X)
    shifts, n_shifts = _check_shifts(shifts)

    n_times, n_chans, n_trials = theshapeof(X)
    X = unsqueeze(X)

    if not assume_centered:
        X = X - X.sum(0, keepdims=1) / n_chans

    if weights.any():  # weights
        X = np.einsum('ijk,ilk->ijk', X, weights)  # element-wise mult
        weights = weights[:n_times, :, :]
        tw = np.sum(weights[:])
    else:  # no weights
        N = 0
        if len(shifts[shifts < 0]):
            N -= np.min(shifts)
        if len(shifts[shifts >= 0]):
            N += np.max(shifts)
        tw = (n_chans * n_shifts - N) * n_trials

    C = np.zeros((n_chans * n_shifts, n_chans * n_shifts))
    for trial in range(n_trials):
        XX = multishift(X[..., trial], shifts)
        XX = XX.reshape(n_times, n_chans * n_shifts)
        C += np.dot(XX.T, XX)

    return C, tw


def _check_weights(weights, X):
    """Check weights dimensions against X."""
    if not isinstance(weights, (np.ndarray, list)):
        if weights is not None:
            warnings.warn('weights should be a list or a numpy array.')
        weights = np.array([])

    if len(weights) > 0:
        dtype = np.complex128 if np.any(np.iscomplex(weights)) else np.float64
        weights = np.asanyarray(weights, dtype=dtype)
        if weights.ndim > 3:
            raise ValueError('Weights must be 3D at most')
        if weights.ndim != X.ndim:
            raise ValueError("Weights array should be the same ndim as X.")
        if weights.shape[1] > 1:
            raise ValueError("Weights array should have a single column.")
        weights = unsqueeze(weights)

    return weights
