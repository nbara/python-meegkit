import numpy as np

from .matrix import multishift, theshapeof, fold, unfold, unsqueeze, shift


def cov_lags(x, y, lags):
    """Empirical covariance of [x,y] with lags.

    [C,tw,m]=nt_cov_lags(x,y,lags)

    C: covariance matrix (3D if length(lags)>1)
    tw: total weight
    m: number of columns in x

    x,y: data matrices
    lags: positive lag means y is delayed relative to x.

    x and y can be time X channels or time X channels X trials.  They can
    also be cell arrays of time X channels matrices.

    See Also
    --------
    nt_relshift

    """
    # if nargin<2 error('!') end
    # if nargin<3 || isempty(lags) lags=[0] end
    # if size(y,1)~=size(x,1) error('!') end
    # if size(y,3)~=size(x,3) error('!') end

    n = x.shape[1] + y.shape[1]  # sum of channels of x and y
    C = np.zeros(n, n, len(lags))
    for t in np.arange(x.shape[2]):
        for l in np.arange(len(lags)):
            yy = shift(y, lags(l), axis=0)
            xy = np.vstack(x, yy)
            C[:, :, l] += xy.T * xy

    m = x.shape[1]
    tw = x.shape[0] * x.shape[2]

    return C, tw, m


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
