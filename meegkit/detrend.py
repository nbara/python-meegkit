"""Robust detrending."""
import numpy as np

from .utils import demean, mrdivide, pca, unfold
from .utils.matrix import _check_weights


def detrend(x, order, w=None, basis='polynomials', threshold=3, n_iter=4,
            show=False):
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
        Raw data matrix.
    order : int
        Order of polynomial or number of sin/cosine pairs
    w : weights, shape=(n_times[, n_channels][, n_trials])
        Sample weights for the regression. If a single channel is provided, the
        same weights are applied to all channels.
    basis : {'polynomials', 'sinusoids'} | ndarray
        Basis for regression.
    threshold : int
        Threshold for outliers, in number of standard deviations (default=3).
    niter : int
        Number of iterations (default=5).

    Returns
    -------
    y : array, shape=(n_times, n_channels[, n_trials])
        Detrended data.
    w : array, shape=(n_times[, n_channels][, n_trials])
        Updated weights.
    r : array, shape=(n_times * ntrials, order)
        Basis matrix used.

    Examples
    --------
    Fit a linear trend, ignoring samples > 3*sd from it, and remove it:
    >> y = detrend(x, 1)

    Fit/remove polynomial (order=5) with initial weighting w, threshold = 4*sd:
    >> y = detrend(x, 5, w, 'polynomial', 4)

    Fit/remove linear then 3rd order polynomial trend:
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
    for i in range(n_iter):
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

    if show:
        _plot_detrend(x, y, w)

    return y, w, r


def regress(x, r, w=None, threshold=1e-7, return_mean=False):
    """Weighted regression.

    Parameters
    ----------
    x : array, shape=(n_times, n_chans)
        Data.
    r : array, shape=(n_times, n_chans)
        Regressor.
    w : array, shape=(n_times, n_chans)
        Weight to apply to `x`. `w` is either a matrix of same size as `x`, or
        a column vector to be applied to each column of `x`.
    threshold : float
        PCA threshold (default=1e-7).
    return_mean : bool
        If True, also return the signal mean prior to regression.

    Returns
    -------
    b : array, shape=(n_chans, n_chans)
        Regression matrix (apply to r to approximate x).
    z : array, shape=(n_times, n_chans)
        Regression (r @ b).

    """
    # check/fix sizes
    w = _check_weights(w, x)
    n_times = x.shape[0]
    n_chans = x.shape[1]
    r = unfold(r)
    x = unfold(x)
    if r.shape[0] != x.shape[0]:
        raise ValueError('r and x have incompatible shapes!')

    # save weighted mean
    mn = x - demean(x, w)

    if not w.any():  # simple regression
        rr = demean(r)
        yy = demean(x)

        # PCA
        V, _ = pca(rr.T.dot(rr), thresh=threshold)
        rrr = rr.dot(V)

        # Regression (OLS)
        b = mrdivide(yy.T.dot(rrr), rrr.T.dot(rrr))
        b = b.T
        z = np.dot(demean(r, w).dot(V), b)
        z = z + mn

    else:  # weighted regression
        if w.shape[0] != n_times:
            raise ValueError('!')

        if w.shape[1] == 1:  # same weight for all channels
            if sum(w.flatten()) == 0:
                print('weights all zero')
                b = 0
            else:
                yy = demean(x, w) * w
                rr = demean(r, w) * w
                V, _ = pca(rr.T.dot(rr), thresh=threshold)
                rrr = rr.dot(V)
                b = mrdivide(yy.T.dot(rrr), rrr.T.dot(rrr))

            z = demean(r, w).dot(V).dot(b.T)
            z = z + mn

        else:  # each channel has own weight
            if w.shape[1] != x.shape[1]:
                raise ValueError('!')
            z = np.zeros(x.shape)
            b = np.zeros((n_chans, n_chans))
            for i in range(n_chans):
                if sum(w[:, i]) == 0:
                    print('weights all zero for channel {}'.format(i))
                    c = np.zeros(x.shape[1], 1)
                else:
                    wc = w[:, i][:, None]  # channel-specific weight
                    yy = demean(x[:, i], wc) * wc
                    # remove channel-specific-weighted mean from regressor
                    r = demean(r, wc)
                    rr = r * wc
                    V, _ = pca(rr.T.dot(rr), thresh=threshold)
                    rr = rr.dot(V)
                    c = mrdivide(yy.T.dot(rr), rr.T.dot(rr))

                z[:, i] = r.dot(V.dot(c.T)).flatten()
                z[:, i] += mn[:, i]
                b[i] = c
            b = b[:, :V.shape[1]]

    return b, z


def _plot_detrend(x, y, w):
    """Plot detrending results."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_times = x.shape[0]
    n_chans = x.shape[1]

    f = plt.figure()
    gs = GridSpec(4, 1, figure=f)
    ax1 = f.add_subplot(gs[:3, 0])
    plt.plot(x, label='original', color='C0')
    plt.plot(y, label='detrended', color='C1')
    ax1.set_xlim(0, n_times)
    ax1.set_xticklabels('')
    ax1.set_title('Robust detrending')
    ax1.legend()

    ax2 = f.add_subplot(gs[3, 0])
    ax2.imshow(w.T, aspect='auto', cmap='Greys')
    ax2.set_yticks(np.arange(1, n_chans + 1, 1))
    ax2.set_xlim(0, n_times)
    ax2.set_ylabel('ch. weights')
    ax2.set_xlabel('samples')
    plt.show()
