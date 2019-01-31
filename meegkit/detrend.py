"""Robust detrending."""
import numpy as np

from scipy.linalg import lstsq, solve

from .utils import demean, pca, unfold
from .utils.matrix import _check_weights


def detrend(x, order, w=None, basis='polynomials', threshold=3, n_iter=4):
    """Robustly remove trend.

    The data are fit to the basis using weighted least squares. The weight is
    updated by setting samples for which the residual is greater than 'thresh'
    times its std to zero, and the fit is repeated at most 'niter'-1 times.

    The choice of order (and basis) determines what complexity of the trend
    that can be removed.  It may be useful to first detrend with a low order
    to avoid fitting outliers, and then increase the order.

    Parameters
    ----------
    x : array, shape=(n_times, n_channels[, n_trials])
        Raw data
    order : int
        Order of polynomial or number of sin/cosine pairs
    w: weights
    basis: {'polynomials', 'sinusoids'} | ndarray
        Basis for regression.
    threshold : int
        Threshold for outliers, in number of standard deviations (default=3).
    niter : int
        Number of iterations (default=5).

    Returns
    -------
    y: detrended data
    w: updated weights
    r: basis matrix used

    Examples
    --------
    Fit linear trend, ignoring samples > 3*sd from it, and remove:
    >> y = detrend(x, 1)

    Fit/remove polynomial order=5 with initial weighting w, threshold = 4*sd:
    >> y = detrend(x, 5, w, [],4 )

    Fit/remove linear then 3rd order polynomial:
    >> [y, w]= detrend(x, 1)
    >> [yy, ww] = detrend(y, 3)

    """
    if threshold == 0:
        raise ValueError('thresh=0 is not what you want...')

    # check/fix sizes
    dims = x.shape
    w = _check_weights(w, x)
    x = unfold(x)
    w = unfold(w)
    n_times, n_chans = x.shape

    # regressors
    if isinstance(basis, np.ndarray):
        r = basis
    else:
        lin = np.linspace(-1, 1, n_times)
        if basis == 'polynomials' or basis is None:
            r = np.zeros((n_times, order))
            for i, o in enumerate(range(1, order + 1)):
                r[:, i] = lin ** o
        elif basis == 'sinusoids':
            r = np.zeros((n_times, order * 2))
            for i, o in enumerate(range(1, order + 1)):
                r[:, 2 * i] = np.sin[2 * np.pi * o * lin / 2]
                r[:, 2 * i + 1] = np.cos[2 * np.pi * o * lin / 2]
        else:
            raise ValueError('!')

    # iteratively remove trends
    # the tricky bit is to ensure that weighted means are removed before
    # calculating the regression (see regress()).
    for iIter in range(n_iter):
        # weighted regression on basis
        _, y = regress(x, r, w)

        # find outliers
        d = x - y
        if w.any():
            d = d * w
        ww = np.ones_like(x)
        ww[(abs(d) > threshold * np.std(d))] = 0

        # update weights
        if not w.any():
            w = ww
        else:
            w = np.amin((w, ww), axis=0)
        del ww

    y = x - y
    y = np.reshape(y, dims)
    w = np.reshape(w, dims)

    # if show: # don't return, just plot
    #     figure(1)
    #     subplot 411
    #     plot(x)
    #     title('raw')
    #     subplot 412
    #     plot(y)
    #     title('detrended')
    #     subplot 413
    #     plot(x-y)
    #     title('trend')
    #     subplot 414
    #     nt_imagescc(w')
    #     title('weight')

    return y, w, r


def regress(y, x, w=None, threshold=1e-7, return_mean=False):
    """Weighted regression.

    Parameters
    ----------
    y : array, shape=(n_times, n_chans)
        Data.
    x : array, shape=(n_times, n_chans)
        Regressor.
    w : array, shape=(n_times, n_chans)
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
        b = mrdivide(yy.T.dot(xxx), xxx.T.dot(xxx))
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
                b = mrdivide(yy.T.dot(xxx), xxx.T.dot(xxx))

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
                    c = mrdivide(yy.T.dot(xx), xx.T.dot(xx))

                z[:, i] = x.dot(V.dot(c.T)).flatten()
                z[:, i] += mn[:, i]
                b[i] = c
            b = b[:, :V.shape[1]]

    return b, z


def mrdivide(A, B):
    r"""Matrix right-division (A/B).

    Solves the linear system XB = A for X. We can write equivalently:

    1) XB = A
    2) (XB).T = A.T
    3) B.T X.T = A.T

    Therefore A/B amounts to solving B.T X.T = A.T for X.T:

    >> mldivide(B.T, A.T).T

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    return mldivide(B.T, A.T).T


def mldivide(A, B):
    r"""Matrix left-division (A\B).

    Solves the AX = B for X. In other words, X minimizes norm(A*X - B), the
    length of the vector AX - B:
    - linalg.solve(A, B) if A is square
    - linalg.lstsq(A, B) otherwise

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    if A.shape[0] == A.shape[1]:
        return solve(A, B)
    else:
        return lstsq(A, B)
