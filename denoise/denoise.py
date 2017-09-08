from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt


def multishift(data, shifts, amplitudes=array([])):
    """Apply multiple shifts to an array."""
    # print "multishift"
    if min(shifts) > 0:
        raise Exception('shifts should be non-negative')

    shifts = shifts.T
    shifts_length = shifts.size

    # array of shift indices
    N = data.shape[0] - max(shifts)
    shiftarray = ((ones((N, shifts_length), int) * shifts).T + r_[0:N]).T
    time, channels, trials = theshapeof(data)
    z = zeros((N, channels * shifts_length, trials))

    if amplitudes:
        for trial in arange(trials):
            for channel in arange(channels):
                y = data[:, channel]
                a = channel * shifts_length
                b = channel * shifts_length + shifts_length
                z[:, arange(a, b), trial] = (y[shiftarray].T * amplitudes).T
    else:
        for trial in xrange(trials):
            for channel in xrange(channels):
                y = data[:, channel]
                a = channel * shifts_length
                b = channel * shifts_length + shifts_length
                z[:, arange(a, b), trial] = y[shiftarray]

    return z


def pcarot(cov, keep=None):
    """PCA rotation from covariance.

    Parameters
    ----------
    cov:  covariance matrix
    keep: number of components to keep [default: all]

    Returns
    --------
    topcs:       PCA rotation matrix
    eigenvalues: PCA eigenvalues
    """

    if not keep:
        keep = cov.shape[0]  # keep all components

    print("cov shape", cov.shape)
    eigenvalues, eigenvector = linalg.eig(cov)

    eigenvalues = eigenvalues.real
    eigenvector = eigenvector.real

    idx = argsort(eigenvalues)[::-1]  # reverse sort ev order
    # eigenvalues = sort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[idx]

    eigenvalues = eigenvalues[::-1]
    idx = idx[::-1]

    topcs = eigenvector[:, idx]

    topcs = topcs[:, arange(keep)]
    eigenvalues = eigenvalues[arange(keep)]

    return topcs, eigenvalues


def tscov(data, shifts=None, w=None):
    """Time shift covariance.

    This function calculates, for each pair [DATA[i], DATA[j]] of columns of
    DATA, the cross-covariance matrix between the time-shifted versions of
    DATA[i]. Shifts are taken from array SHIFTS. Weights are taken from `w`.

    DATA can be 1D, 2D or 3D.  WEIGHTS is 1D (if DATA is 1D or 2D) or
    2D (if DATA is 3D).

    Output is a 2D matrix with dimensions (ncols(X)*nshifts)^2.
    This matrix is made up of a DATA.shape[1]^2 matrix of submatrices
    of dimensions nshifts**2.

    The weights are not shifted.

    Parameters
    ----------
    data: array
        Data.
    shifts: array
        Array of time shifts (must be non-negative).
    w: array
        Weights.

    Returns
    -------
    covariance_matrix: array
        Covariance matrix.
    total_weight: array
        Total weight (covariance_matrix/total_weight is normalized
        covariance).
    """

    if shifts is None:
        shifts = array([0])
    if not any(w):
        w = array([])

    if shifts.min() < 0:
        raise ValueError("Shifts should be non-negative.")

    nshifts = shifts.size

    samples, channels, trials = theshapeof(data)
    covariance_matrix = zeros((channels * nshifts, channels * nshifts))

    if any(w):
        # weights
        if w.shape[1] > 1:
            raise ValueError("Weights array should have a single column.")

        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = multishift(data[:, :, trial], shifts)
            elif data.ndim == 2:
                data = unsqueeze(data)
                shifted_trial = multishift(data[:, trial], shifts)
            else:
                data = unsqueeze(data)
                shifted_trial = multishift(data[trial], shifts)

            trial_weight = w[arange(shifted_trial.shape[0]), :, trial]
            shifted_trial = (squeeze(shifted_trial).T *
                             squeeze(trial_weight)).T
            covariance_matrix += dot(shifted_trial.T, shifted_trial)

        total_weight = sum(w[:])
    else:
        # no weights
        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = squeeze(multishift(data[:, :, trial], shifts))
            else:
                shifted_trial = multishift(data[:, trial], shifts)

            covariance_matrix += dot(shifted_trial.T, shifted_trial)

        total_weight = shifted_trial.shape[0] * trials

    return covariance_matrix, total_weight


def fold(data, epochsize):
    """Fold 2D data into 3D."""
    return transpose(reshape(data, (epochsize, data.shape[0] /
                                    epochsize, data.shape[1]),
                             order="F").copy(), (0, 2, 1))


def unfold(data):
    """Unfold 3D data."""

    samples, channels, trials = theshapeof(data)

    if trials > 1:
        return reshape(transpose(data, (0, 2, 1)),
                       (samples * trials, channels), order="F").copy()
    else:
        return data


def theshapeof(data):
    """docstring for theshape"""
    if data.ndim == 3:
        return data.shape[0], data.shape[1], data.shape[2]
    elif data.ndim == 2:
        return data.shape[0], data.shape[1], 1
    elif data.ndim == 1:
        return data.shape[0], 1, 1
    else:
        raise ValueError("Array contains more than 3 dimensions")


def demean(data, w=None):
    """Remove weighted mean over columns."""

    samples, channels, trials = theshapeof(data)

    data = unfold(data)

    if any(w):
        w = unfold(w)

        if w.shape[0] != data.shape[0]:
            raise ValueError, "Data and weights arrays should have same number of rows and pages."

        if w.shape[1] == 1 or w.shape[1] == channels:
            the_mean = sum(data * w) / sum(w)
        else:
            raise ValueError, "Weight array should have either the same number of columns as data array, or 1 column."

        demeaned_data = data - the_mean
    else:
        the_mean = mean(data, 0)
        demeaned_data = data - the_mean

    demeaned_data = fold(demeaned_data, samples)

    #the_mean.shape = (1, the_mean.shape[0])

    return demeaned_data, the_mean


def normcol(data, w=None):
    """
    Normalize each column so its weighted msq is 1.

    If DATA is 3D, pages are concatenated vertically before calculating the
    norm.

    Weight should be either a column vector, or a matrix (2D or 3D) of same
    size as data.

    Parameters
    ----------
    data: data to normalize
    w: weight

    Returns
    --------
    normalized_data: normalized data

    """

    if data.ndim == 3:
        samples, channels, trials = data.shape
        data = unfold(data)
        if not w.any():
            # no weights
            normalized_data = fold(normcol(data), samples)
        else:
            if w.shape[0] != samples:
                raise ValueError("Weight array should have same number of' \
                                 'columns as data array.")

            if w.ndim == 2 and w.shape[1] == 1:
                w = tile(w, (1, samples, trials))

            if w.shape != w.shape:
                raise ValueError("Weight array should have be same shape ' \
                                 'as data array")

            w = unfold(w)

            normalized_data = fold(normcol(data, w), samples)
    else:
        samples, channels = data.shape
        if not w.any():
            normalized_data = data * ((sum(data ** 2) / samples) ** -0.5)
        else:
            if w.shape[0] != data.shape[0]:
                raise ValueError(
                    "Weight array should have same number of columns as data array.")

            if w.ndim == 2 and w.shape[1] == 1:
                w = tile(w, (1, channels))

            if w.shape != data.shape:
                raise ValueError(
                    "Weight array should have be same shape as data array")

            if w.shape[1] == 1:
                w = tile(w, (1, channels))

            normalized_data = data * \
                (sum((data ** 2) * w) / sum(w)) ** -0.5

    return normalized_data


def regcov(cxy, cyy, keep=array([]), threshold=array([])):
    """regression matrix from cross covariance"""

    # PCA of regressor
    [topcs, eigenvalues] = pcarot(cyy)

    # discard negligible regressor PCs
    if keep:
        keep = max(keep, topcs.shape[1])
        topcs = topcs[:, 0:keep]
        eigenvalues = eigenvalues[0:keep]

    if threshold:
        idx = where(eigenvalues / max(eigenvalues) > threshold)
        topcs = topcs[:, idx]
        eigenvalues = eigenvalues[idx]

    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = dot(topcs.T, cxy)

    # projection matrix from regressor PCs
    r = (r.T * 1 / eigenvalues).T

    # projection matrix from regressors
    r = dot(squeeze(topcs), squeeze(r))

    return r


def tsxcov(x, y, shifts=None, w=array([])):
    """Calculate cross-covariance of X and time-shifted Y.

    This function calculates, for each pair of columns (Xi,Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj.
    Shifts are taken from array SHIFTS.

    The weights are applied to X.

    X can be 1D, 2D or 3D.  W is 1D (if X is 1D or 2D) or 2D (if X is 3D).

    Output is a 2D matrix with dimensions ncols(X)*(ncols(Y)*nshifts).

    Parameters
    ----------
    x, y: arrays
        data to cross correlate
    shifts: array
        time shifts (must be non-negative)
    w: array
        weights

    Returns
    -------
    c: cross-covariance matrix
    tw: total weight
    """

    if shifts == None:
        shifts = array([0])

    nshifts = shifts.size

    mx, nx, ox = x.shape
    my, ny, oy = y.shape
    c = zeros((nx, ny * nshifts))

    if any(w):
        x = fold(unfold(x) * unfold(w), mx)

    # cross covariance
    for trial in xrange(ox):
        yy = squeeze(multishift(y[:, :, trial], shifts))
        xx = squeeze(x[0:yy.shape[0], :, trial])

        c += dot(xx.T, yy)

    if not any(w):
        tw = ox * ny * yy.shape[0]
    else:
        w = w[0:yy.shape[0], :, :]
        tw = sum(w[:])

    return c, tw


def tsregress(x, y, shifts=array([0]), keep=array([]), threshold=array([]),
              toobig1=array([]), toobig2=array([])):
    """docstring for tsregress"""

    # shifts must be non-negative
    mn = shifts.min()
    if mn < 0:
        shifts = shifts - mn
        x = x[-mn + 1:, :, :]
        y = y[-mn + 1:, :, :]

    nshifts = shifts.size

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
    [cxy, totalweight] = tscov2(x, y, shifts.T, xw, yw)
    cxy = cxy / totalweight

    # regression matrix
    r = regcov(cxy, cyy, keep, threshold)

    # regression
    if x.ndim == 3:
        x = unfold(x)
        y = unfold(y)

        [m, n, o] = x.shape
        mm = m - max(shifts)
        z = zeros(x.shape)

        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk + 1:kk + mm]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]

        z = fold(z, Mx)
        z = z[0:-max(shifts), :, :]
    else:
        m, n = x.shape
        z = zeros((m - max(shifts), n))
        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk + 1:kk + z.shape[0]]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z = z + y[idx1, :] * r[idx2, :]

    offset = max(0, -mn)
    idx = r_[offset + 1:offset + z.shape[0]]

    return [z, idx]


def find_outliers(x, toobig1, toobig2=[]):
    """docstring for find_outliers"""

    [m, n, o] = x.shape
    x = unfold(x)

    # remove mean
    x = demean(x)[0]

    # apply absolute threshold
    w = ones(x.shape)
    if toobig1:
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]

        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]

        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
    else:
        w = ones(x.shape)

    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, w)
        X = tile(X, (x.shape[0], 1))
        idx = where(x**2 > (X * toobig2))
        w[idx] = 0

    w = fold(w, m)

    return w


def find_outlier_trials(x, thresh=None, disp_flag=True):
    """Find outlier trials.

    For example thresh=2 rejects trials that deviate from the mean by
    more than twice the average deviation from the mean.

    Parameters
    ----------
    x : ndarray
        Data array (trials * channels * time).
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

    if len(x.shape) > 3:
        raise ValueError('x should be 2D or 3D')
    elif len(x.shape) == 3:
        n, c, t = x.shape  # trials * channels * time
        x = np.reshape(x, (n, c * t))
    else:
        n, _ = x.shape

    m = np.mean(x, axis=0)  # mean over trials
    m = np.tile(m, (n, 1))  # repeat mean
    d = x - m  # difference from mean
    dd = np.zeros(n)
    for i_trial in range(n):
        dd[i_trial] = np.sum(d[i_trial, :] ** 2)
    d = dd / (np.sum(x.flatten() ** 2) / n)
    idx = np.where(d < thresh[0])[0]
    del dd

    if disp_flag:
        plt.figure(figsize=(7, 4))
        gs = gridspec.GridSpec(1, 2)

        plt.suptitle('Outlier trial detection')

        # Before
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(d, ls='-')
        ax1.plot(np.setdiff1d(range(n), idx),
                 d[np.setdiff1d(range(n), idx)], color='r', ls=' ', marker='.')
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
    if len(idx) < n:
        bads = np.setdiff1d(range(n), idx)

    return bads, d


def wmean(x, w=[], dim=0):
    """docstring for wmean"""

    if not w:
        y = mean(x, dim)
    else:
        if x.shape[0] != w.shape[0]:
            raise Exception("data and weight must have same nrows")
        if w.shape[1] == 1:
            w = tile(w, (1, x.shape(1)))
        if w.shape[1] != x.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")

        y = sum(x * w, dim) / sum(w, dim)

    return y


def mean_over_trials(x, w):
    """docstring for mean_over_trials"""

    m, n, o = x.shape

    if not any(w):
        y = mean(x, 2)
        tw = ones((m, n, 1)) * o
    else:
        mw, nw, ow = w.shape
        if mw != m:
            raise "!"
        if ow != o:
            raise "!"

        x = unfold(x)
        w = unfold(w)

        if nw == n:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) / sum(w, 3)
        elif nw == 1:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) * 1 / sum(w, 3)

        tw = sum(w, 3)

    return y, tw


def wpwr(x, w=None):
    """Weighted power."""

    if w is None:
        w = array([])

    x = unfold(x)
    w = unfold(w)

    if w:
        x = x * w
        y = sum(x ** 2)
        tweight = sum(w)
    else:
        y = sum(x ** 2)
        tweight = x.size

    return y, tweight


def unsqueeze(data):
    """Add singleton dimensions to an array."""

    return data.reshape(theshapeof(data))
