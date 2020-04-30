"""Utils for ASR functions."""
import numpy as np
from scipy.special import gamma, gammaincinv
from numpy import linalg
from numpy.matlib import repmat
from scipy import signal
from scipy.linalg import toeplitz


def fit_eeg_distribution(X, min_clean_fraction=0.25, max_dropout_fraction=0.1,
                         fit_quantiles=[0.022, 0.6],
                         step_sizes=[0.0220, 0.6000],
                         shape_range=np.linspace(1.7, 3.5, 13)):
    """Estimate the mean and SD of clean EEG from contaminated data.

    This function estimates the mean and standard deviation of clean EEG from a
    sample of amplitude values (that have preferably been computed over short
    windows) that may include a large fraction of contaminated samples. The
    clean EEG is assumed to represent a generalized Gaussian component in a
    mixture with near-arbitrary artifact components. By default, at least 25%
    (``min_clean_fraction``) of the data must be clean EEG, and the rest can be
    contaminated. No more than 10% (``max_dropout_fraction``) of the data is
    allowed to come from contaminations that cause lower-than-EEG amplitudes
    (e.g., sensor unplugged). There are no restrictions on artifacts causing
    larger-than-EEG amplitudes, i.e., virtually anything is handled (with the
    exception of a very unlikely type of distribution that combines with the
    clean EEG samples into a larger symmetric generalized Gaussian peak and
    thereby "fools" the estimator). The default parameters should work for a
    wide range of applications but may be adapted to accommodate special
    circumstances.

    The method works by fitting a truncated generalized Gaussian whose
    parameters are constrained by ``min_clean_fraction``,
    ``max_dropout_fraction``, ``fit_quantiles``, and ``shape_range``. The fit
    is performed by a grid search that always finds a close-to-optimal solution
    if the above assumptions are fulfilled.

    Parameters
    ----------
    X : array, shape=(n_channels, n_samples)
        EEG data, possibly containing artifacts.
    max_dropout_fraction : float
        Maximum fraction that can have dropouts. This is the maximum fraction
        of time windows that may have arbitrarily low amplitude (e.g., due to
        the sensors being unplugged) (default=0.25).
    min_clean_fraction : float
        Minimum fraction that needs to be clean. This is the minimum fraction
        of time windows that need to contain essentially uncontaminated EEG
        (default=0.1).
    fit_quantiles : 2-tuple
        Quantile range [lower,upper] of the truncated generalized Gaussian
        distribution that shall be fit to the EEG contents (default=[0.022
        0.6]).
    step_sizes : 2-tuple
        Step size of the grid search; the first value is the stepping of the
        lower bound (which essentially steps over any dropout samples), and the
        second value is the stepping over possible scales (i.e., clean-data
        quantiles) (default=[0.01, 0.01]).
    beta : array
        Range that the clean EEG distribution's shape parameter beta may take.

    Returns
    -------
    mu : array
        Estimated mean of the clean EEG distribution.
    sig : array
        Estimated standard deviation of the clean EEG distribution.
    alpha : float
        Estimated scale parameter of the generalized Gaussian clean EEG
        distribution.
    beta : float
        Estimated shape parameter of the generalized Gaussian clean EEG
        distribution.

    """
    # sort data so we can access quantiles directly
    X = np.sort(X)
    n = len(X)

    # calc z bounds for the truncated standard generalized Gaussian pdf and
    # pdf rescaler
    quants = np.array(fit_quantiles)
    zbounds = []
    rescale = []
    for b in range(len(shape_range)):
        gam = gammaincinv(
            1 / shape_range[b], np.sign(quants - 1 / 2) * (2 * quants - 1))
        zbounds.append(np.sign(quants - 1 / 2) * gam ** (1 / shape_range[b]))
        rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

    # determine the quantile-dependent limits for the grid search
    # we can generally skip the tail below the lower quantile
    lower_min = np.min(quants)
    # maximum width is the fit interval if all data is clean
    max_width = np.diff(quants)
    # minimum width of the fit interval, as fraction of data
    min_width = min_clean_fraction * max_width

    rowval = np.array(
        np.round(n * np.arange(lower_min, lower_min +
                               max_dropout_fraction + (step_sizes[0] * 1e-9),
                               step_sizes[0])))
    colval = np.array(np.arange(0, int(np.round(n * max_width))))
    newX = []
    for iX in range(len(colval)):
        newX.append(X[np.int_(iX + rowval)])

    X1 = newX[0]
    newX = newX - repmat(X1, len(colval), 1)
    opt_val = np.inf

    for m in (np.round(n * np.arange(max_width, min_width, -step_sizes[1]))):
        mcurr = int(m - 1)
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        rowval = np.array(nbins / newX[mcurr])
        H = newX[0:int(m)] * repmat(rowval, int(m), 1)

        hist_all = []
        for ih in range(len(rowval)):
            histcurr = np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))
            hist_all.append(histcurr[0])
        hist_all = np.array(hist_all, dtype=int).T
        hist_all = np.vstack((hist_all, np.zeros(len(rowval), dtype=int)))
        logq = np.log(hist_all + 0.01)

        # for each shape value...
        for b in range(len(shape_range)):
            bounds = zbounds[b]
            x = bounds[0] + (np.arange(0.5, nbins + 0.5) /
                             nbins * np.diff(bounds))
            p = np.exp(-np.abs(x)**shape_range[b]) * rescale[b]
            p = p / np.sum(p)

            # calc KL divergences
            kl = np.sum(
                np.transpose(repmat(p, logq.shape[1], 1)) *
                (np.transpose(repmat(np.log(p), logq.shape[1], 1)) -
                 logq[:-1, :]),
                axis=0) + np.log(m)

            # update optimal parameters
            min_val = np.min(kl)
            idx = np.argmin(kl)

            if (min_val < opt_val):
                opt_val = min_val
                opt_beta = shape_range[b]
                opt_bounds = bounds
                opt_lu = [X1[idx], (X1[idx] + newX[int(m - 1), idx])]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sig = np.sqrt((alpha**2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sig, alpha, beta


def yulewalk(order, F, M):
    """Recursive filter design using a least-squares method.

    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:

    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)

    matches the magnitude frequency response given by vectors F and M.

    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.

    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed explanation
    of the algorithm see [1]_.

    Parameters
    ----------
    order : int
        Filter order.
    F : array
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : array
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.

    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace Electronic
           Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.

    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:

    >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    >>> [b, a] = yulewalk(8, f, m) # Filter design using a least-squares method

    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate(
        (1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))),
        axis=None)
    A = polystab(denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(signal.freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(
        np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=np.complex))))
    )
    B = np.real(numf(hh[0:nr], A, nb))

    return B, A


def yulewalk_filter(X, sfreq, zi=None, ab=None, axis=-1):
    """Yulewalk filter.

    Parameters
    ----------
    X : array, shape = (n_channels, n_samples)
        Data to filter.
    sfreq : float
        Sampling frequency.
    zi : array, shape=(n_channels, filter_order)
        Initial conditions.
    a, b : 2-tuple | None
        Coefficients of an IIR filter that is used to shape the spectrum of the
        signal when calculating artifact statistics. The output signal does not
        go through this filter. This is an optional way to tune the sensitivity
        of the algorithm to each frequency component of the signal. The default
        filter is less sensitive at alpha and beta frequencies and more
        sensitive at delta (blinks) and gamma (muscle) frequencies.
    axis : int
        Axis to filter on (default=-1, corresponding to samples).

    Returns
    -------
    out : array
        Filtered data.
    zf :  array, shape=(n_channels, filter_order)
        Output filter state.

    """
    [C, S] = X.shape
    if ab is None:
        F = np.array([0, 2, 3, 13, 16, 40, np.minimum(
            80.0, (sfreq / 2.0) - 1.0), sfreq / 2.0]) * 2.0 / sfreq
        M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
        B, A = yulewalk(8, F, M)
    else:
        A, B = ab

    # apply the signal shaping filter and initialize the IIR filter state
    if zi is None:
        zi = signal.lfilter_zi(B, A)
        out, zf = signal.lfilter(B, A, X, zi=zi[:, None] * X[:, 0], axis=axis)
    else:
        out, zf = signal.lfilter(B, A, X, zi=zi, axis=axis)

    return out, zf


def block_geometric_median(X, blocksize, tol=1e-5, max_iter=500):
    """Calculate a blockwise geometric median.

    This is faster and less memory-intensive than the regular geom_median
    function. This statistic is not robust to artifacts that persist over a
    duration that is significantly shorter than the blocksize.

    Parameters
    ----------
    X : array, shape=(observations, variables)
        The data.
    blocksize : int
        The number of successive samples over which a regular mean should be
        taken.
    tol : float
        Tolerance (default=1e-5)
    max_iter : int
        Max number of iterations (default=500).

    Returns
    -------
    g : array,
        Geometric median over X.

    Notes
    -----
    This function is noticeably faster if the length of the data is divisible
    by the block size. Uses the GPU if available.

    """
    if (blocksize > 1):
        o, v = X.shape       # #observations & #variables
        r = np.mod(o, blocksize)  # #rest in last block
        b = int((o - r) / blocksize)   # #blocks
        Xreshape = np.zeros((b + 1, v))
        if (r > 0):
            Xreshape[0:b, :] = np.reshape(
                np.sum(np.reshape(X[0:(o - r), :],
                                  (blocksize, b * v)), axis=0),
                (b, v))
            Xreshape[b, :] = np.sum(
                X[(o - r + 1):o, :], axis=0) * (blocksize / r)
        else:
            Xreshape = np.reshape(
                np.sum(np.reshape(X, (blocksize, b * v)), axis=0), (b, v))
        X = Xreshape

    y = np.median(X, axis=0)
    y = geometric_median(X, tol, y, max_iter) / blocksize

    return y


def geometric_median(X, tol, y, max_iter):
    """Calculate the geometric median for a set of observations.

    This is using Weiszfeld's algorithm (mean under a Laplacian noise
    distribution)

    Parameters
    ----------
    X : the data, as in mean
    tol : tolerance (default=1.e-5)
    y : initial value (default=median(X))
    max_iter : max number of iterations (default=500)

    Returns
    -------
    g : geometric median over X

    """
    for i in range(max_iter):
        invnorms = 1 / np.sqrt(
            np.sum((X - repmat(y, X.shape[0], 1))**2, axis=1))
        oldy = y
        y = np.sum(X * np.transpose(
            repmat(invnorms, X.shape[1], 1)), axis=0
        ) / np.sum(invnorms)

        if ((linalg.norm(y - oldy) / linalg.norm(y)) < tol):
            break

    return y


def moving_average(N, X, Zi):
    """Moving-average filter along the second dimension of the data.

    Parameters
    ----------
    N : filter length in samples
    X : data matrix [#Channels x #Samples]
    Zi : initial filter conditions (default=[])

    Returns
    -------
    X : the filtered data
    Zf : final filter conditions

    Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
    2012-01-10

    """
    [C, S] = X.shape

    if Zi is None:
        Zi = np.zeros((C, N))

    # pre-pend initial state & get dimensions
    Y = np.concatenate((Zi, X), axis=1)
    [CC, M] = Y.shape

    # get alternating index vector (for additions & subtractions)
    idx = np.vstack((np.arange(0, M - N), np.arange(N, M)))

    # get sign vector (also alternating, and includes the scaling)
    S = np.vstack((- np.ones((1, M - N)), np.ones((1, M - N)))) / N

    # run moving average
    YS = np.zeros((C, S.shape[1] * 2))
    for i in range(C):
        YS[i, :] = Y[i, idx.flatten(order='F')] * S.flatten(order='F')

    X = np.cumsum(YS, axis=1)
    # read out result
    X = X[:, 1::2]

    Zf = np.transpose(
        np.vstack((-((X[:, -1] * N) - Y[:, -N])),
                  np.transpose(Y[:, -N + 1:]))
    )

    return X, Zf


def polystab(a):
    """Polynomial stabilization.

    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.

    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.

    >>> h = fir1(25,0.4);               # Window-based FIR filter design
    >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    >>> flag_minphase = isminphase(hmin)# Determines if filter is minimum phase

    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not(np.sum(np.imag(a))):
        b = np.real(b)

    return b


def numf(h, a, nb):
    """Find numerator B given impulse-response h of B/A and denominator A.

    NB is the numerator order.  This function is used by YULEWALK.
    """
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = signal.lfilter(np.array([1.0]), a, xn)

    b = linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b


def denf(R, na):
    """Compute denominator from covariances.

    A = DENF(R,NA) computes order NA denominator A from covariances
    R(0)...R(nr) using the Modified Yule-Walker method. This function is used
    by YULEWALK.

    """
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A
