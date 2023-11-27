"""Audio and signal processing tools."""
import numpy as np
import scipy.signal as ss
from scipy.linalg import lstsq, solve, toeplitz
from scipy.signal import lfilter

from .covariances import convmtx


def modulation_index(phase, amp, n_bins=18):
    """Compute the Modulation Index (MI) between two signals.

    MI is a measure of the amount of phase-amplitude coupling. Phase angles are
    expected to be in radians [1]_. MI is derived from the Kullbach-Leibner
    distance, a measure for the disparity of two distributions, which is also
    returned here. MI is recommend the modulation index for noisy and short
    data epochs with unknown forms of coupling [2]_.

    Parameters
    ----------
    phase : array
        Phase vector, in radians.
    amp : array
        Amplitude vector.
    n_bins : int
        Number of bins in which to discretize phase (default=18 bins, giving
        a 20-degree resolution).

    Returns
    -------
    MI : array
        Tort's Modulation index.
    KL : array
        Kullbach-Leibner distance.

    Examples
    --------
    >> phas = np.random.rand(100, 1) * 2 * np.pi - np.pi
    >> ampl = rng.standard_normal((100, 1) * 30 + 100
    >> MI, KL = modulation_index(phas, ampl)

    Notes
    -----
    Phase and amplitude can be derived directly from any time series through
    the analytic signal:
    >> analytic_signal = hilbert(filtered_data)
    >> phase = np.phase(analytic_signal)
    >> amplitude = np.abs(analytic_signal)

    MI can be subjected to permutation testing to assess significance. For
    permutation testing, the observed coupling value is compared to a
    distribution of shuffled coupling values. Shuffled coupling values are
    constructed by calculating the coupling value between the original phase
    time series and a permuted amplitude time series. The permuted amplitude
    time series can be constructed by cutting the amplitude time series at a
    random time point and reversing the order of both parts [2]_. The observed
    coupling value is standardized to the distribution of the shuffled coupling
    values according to the following formula:

    MI_z = (MI_observed − µ_MI_shuffled) / σ_MI_shuffled

    where μ denotes the mean and σ the standard deviation. Only when the
    observed phase-locking value is larger than 95% of shuffled values, it is
    defined as significant. See [2]_ for details.

    References
    ----------
    .. [1] Tort AB, Komorowski R, Eichenbaum H, Kopell N. Measuring
       phase-amplitude coupling between neuronal oscillations of different
       frequencies. J Neurophysiol. 2010 Aug;104(2):1195-210. doi:
       10.1152/jn.00106.2010.
    .. [2] Huelsemann, M. J., Naumann, E., & Rasch, B. (2018). Quantification
       of Phase-Amplitude Coupling in Neuronal Oscillations: Comparison of
       Phase-Locking Value, Mean Vector Length, and Modulation Index. bioRxiv,
       290361.

    """
    phase = phase.squeeze()
    amp = amp.squeeze()
    if phase.shape != amp.shape or phase.ndims > 1 or amp.ndims:
        raise AttributeError("Inputs must be 1D vectors of same length.")

    # Convert phase to degrees
    phasedeg = np.degrees(phase)

    # Calculate mean amplitude in each phase bin
    binsize = 360 / n_bins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phasedeg >= phase_lo[b],
                                    phasedeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(amp[phaserange])

    # Compute the probability of an amplitude unit being in a phase bin
    p_j = mean_amp / np.sum(mean_amp)

    # Get a meaningful KL distance when observed probability in a bin is 0
    if np.any(p_j == 0):
        p_j[p_j == 0] = np.finfo(float).eps

    # Phase-amplitude coupling is defined by a distribution that significantly
    # deviates from the uniform distribution. Kullback-Leibler distance is
    # calculated by the following formula: KL = log(N) − H(p), where H is
    # Shannon entropy, and N is the number of bins.
    H = -np.sum(p_j * np.log10(p_j))
    Hmax = np.log10(n_bins)
    KL = Hmax - H
    MI = KL / Hmax

    return MI, KL


def smooth(x, window_len, window="square", axis=0, align="left"):
    """Smooth a signal using a window with requested size along a given axis.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with
    the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output signal.

    Parameters
    ----------
    x : array
        The input signal.
    window_len : float
        The dimension of the smoothing window (in samples). Can be fractionary.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' flat window will produce a moving average smoothing.
    axis : int
        Axis along which smoothing will be applied (default=0).
    align : {'left' | 'center'}
        If `left` (default), the convolution if computed in a causal way by
        shifting the output of a normal convolution by the kernel size. If
        `center`, the center of the impulse is used as the location where the
        convolution is summed.

    Returns
    -------
    y : array
        The smoothed signal.

    Examples
    --------
    >> t = linspace(-2, 2, 0.1)
    >> x = sin(t) + randn(len(t)) * 0.1
    >> y = smooth(x, 2)

    See Also
    --------
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve,
    scipy.signal.lfilter

    Notes
    -----
    length(output) != length(input), to correct this, we return :
    >> y[(window_len / 2 - 1):-(window_len / 2)]  # noqa
    instead of just y.

    """
    if x.shape[axis] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window not in ["square", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Unknown window type.")
    if window_len == 0:
        raise ValueError("Smoothing kernel must be at least 1 sample wide")
    if window_len == 1:
        return x

    def _smooth1d(x, n, align="left"):
        if x.ndim != 1:
            raise ValueError("Smooth only accepts 1D arrays")

        frac, n = np.modf(n)
        n = int(n)

        if window == "square":  # moving average
            w = np.ones(n, "d")
            w = np.r_[w, frac]
        else:
            w = eval("np." + window + "(n)")

        if align == "center":
            a = x[n - 1:0:-1]
            b = x[-2:-n - 1:-1]
            s = np.r_[a, x, b]
            out = np.convolve(w / w.sum(), s, mode="same")
            return out[len(a):-len(b)]

        elif align == "left":
            out = ss.lfilter(w / w.sum(), 1, x)
            return out

    if x.ndim > 1:  # apply along given axis
        y = np.apply_along_axis(_smooth1d, axis, x, n=window_len)
    else:
        y = _smooth1d(x, n=np.abs(window_len))

    return y


def lowpass_env_filtering(x, cutoff=150., n=1, sfreq=22050):
    """Low-pass filters a signal using a Butterworth filter.

    Parameters
    ----------
    x : ndarray
    cutoff : float, optional
        Cut-off frequency of the low-pass filter, in Hz. The default is 150 Hz.
    n : int, optional
        Order of the low-pass filter. The default is 1.
    sfreq : float, optional
        Sampling frequency of the signal to filter. The default is 22050 Hz.

    Returns
    -------
    ndarray
        Low-pass filtered signal.

    """
    b, a = ss.butter(N=n, Wn=cutoff * 2. / sfreq, btype="lowpass")
    return ss.lfilter(b, a, x)


def hilbert_envelope(x):
    """Calculate the Hilbert envelope of a signal.

    Parameters
    ----------
    x : array
        Signal on which to calculate the hilbert envelope. The calculation
        is done along the last axis (i.e. ``axis=-1``).

    Returns
    -------
    ndarray

    """
    def next_pow_2(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()

    signal = np.asarray(x)
    N_orig = signal.shape[-1]
    # Next power of 2.
    N = next_pow_2(N_orig)
    y_h = ss.hilbert(signal, N)
    # Return signal with same shape as original
    return np.abs(y_h[..., :N_orig])


def spectral_envelope(x, sfreq, lowpass=32):
    """Compute envelope with convolution.

    Notes
    -----
    The signal is first padded to avoid edge effects. To align the envelope
    with the input signal, we return :
    >> y[(window_len / 2 - 1):-(window_len / 2)]  # noqa

    """
    x = np.squeeze(x)
    if x.ndim > 1:
        raise AttributeError("x must be 1D")
    if lowpass is None:
        lowpass = sfreq / 2

    # Pad signal with reflection
    win = sfreq // lowpass  # window size in samples
    a = x[win - 1:0:-1]
    b = x[-2:-win - 1:-1]
    s = np.r_[a, x, b]

    # Convolve squared signal with a square window and take cubic root
    y = np.convolve(s ** 2, np.ones((win,)) / win, mode="same") ** (1 / 3)
    return y[len(a):-len(b)]


def gaussfilt(data, srate, f, fwhm, n_harm=1, shift=0, return_empvals=False, show=False):
    """Narrow-band filter via frequency-domain Gaussian.

    Empirical frequency and FWHM depend on the sampling rate and the
    number of time points, and may thus be slightly different from
    the requested values.

    Parameters
    ----------
    data : ndarray
        EEG data, shape=(n_samples, n_channels[, ...])
    srate : int
        Sampling rate in Hz.
    f : float
        Break frequency of filter.
    fhwm : float
        Standard deviation of filter, defined as full-width at half-maximum
        in Hz.
    n_harm : int
        Number of harmonics of the frequency to consider.
    shift : int
        Amount shift peak frequency by (only useful when considering harmonics,
        otherwise leave to 0).
    return_empvals : bool
        Return empirical values (default: False).
    show : bool
        Set to True to show the frequency-domain filter shape.

    Returns
    -------
    filtdat : ndarray
        Filtered data.
    empVals : float
        The empirical frequency and FWHM.
    """
    # input check
    assert (data.shape[1] <= data.shape[0]
            ), "n_channels must be less than n_samples"
    assert ((f - fwhm) >= 0), "increase frequency or decrease FWHM"
    assert (fwhm >= 0), "FWHM must be greater than 0"

    # frequencies
    hz = np.fft.fftfreq(data.shape[0], 1. / srate)
    empVals = np.zeros((2,))

    # compute empirical frequency and standard deviation
    idx_p = np.searchsorted(hz[hz >= 0], f, "left")

    # create Gaussian
    fx = np.zeros_like(hz)
    for i_harm in range(1, n_harm + 1):  # make one gaussian per harmonic
        s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
        x = hz.copy()
        x -= (f * i_harm - shift)
        gauss = np.exp(-.5 * (x / s)**2)  # gaussian
        gauss = gauss / np.max(gauss)  # gain-normalized
        fx = fx + gauss

    # create Gaussian
    for i_harm in range(1, n_harm + 1):  # make one gaussian per harmonic
        s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
        x = hz.copy()
        x += (f * i_harm - shift)
        gauss = np.exp(-.5 * (x / s) ** 2)  # gaussian
        gauss = gauss / np.max(gauss)  # gain-normalized
        fx = fx + gauss

    # filter
    if data.ndim == 2:
        filtdat = 2 * np.real(np.fft.ifft(
                              np.fft.fft(data, axis=0) * fx[:, None], axis=0))
    elif data.ndim == 3:
        filtdat = 2 * np.real(np.fft.ifft(
                              np.fft.fft(data, axis=0) * fx[:, None, None],
                              axis=0))

    if return_empvals or show:
        empVals[0] = hz[idx_p]
        # find values closest to .5 after MINUS before the peak
        empVals[1] = hz[idx_p - 1 + np.searchsorted(fx[:idx_p], 0.5)] \
            - hz[np.searchsorted(fx[:idx_p + 1], 0.5)]

    if show:
        # inspect the Gaussian (turned off by default)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(hz, fx, "o-")
        plt.xlim([0, None])

        title = f"Requested: {f}, {fwhm} Hz\nEmpirical: {empVals[0]}, {empVals[1]} Hz"
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude gain")
        plt.show()

    if return_empvals:
        return filtdat, empVals
    else:
        return filtdat


def teager_kaiser(x, m=1, M=1, axis=0):
    """Mean Teager-Kaiser energy operator.

    The discrete version of the Teager-Kaiser operator is computed according
    to:

    y[n] = x[n] ** {2 / m} - (x[n - M] * x[n + M]) ** {1 / m}

    with m the exponent parameter and M the lag parameter which both are
    usually equal to 1 for a conventional operator. The Teaser-Kaiser operator
    can be used to track amplitude modulations (AM) and/or frequency
    modulations (FM).

    Parameters
    ----------
    x : array, shape=(n_samples[, n_channels][, n_trials])
        Input data.
    m : int
        Exponent parameter.
    M : int
        Lag parameter.
    axis : int
        Axis to compute metric on.

    Returns
    -------
    array, shape=(n_samples - 2 * M[, n_channels][, n_trials])
        Instantaneous energy.

    References
    ----------
    Adapted from the TKEO function in R library `seewave`.

    Examples
    --------
    >>> x = np.array([1,  3, 12, 25, 10])
    >>> tk_energy = teager_kaiser(x)

    """
    def tg(x, M, m):
        return x[M:-M] ** (2 / m) - (x[2 * M:] * x[:-2 * M]) ** (1 / m)

    return np.apply_along_axis(tg, axis, x, M, m)


def slope_sum(x, w: int, axis=0):
    r"""Slope sum function.

    The discrete version of the Teager-Kaiser operator is computed according
    to:

    y[n] = \\sum_{i=k-w}^k (y_i -y_{i-1})

    Parameters
    ----------
    x : array, shape=(n_samples[, n_channels][, n_trials])
        Input data.
    w : int
        Window.

    References
    ----------
    https://ieeexplore.ieee.org/document/8527395

    """
    def ss(x, w):
        out = np.diff(x, prepend=0)
        out = out.cumsum()
        out[w:] = out[w:] - out[:-w]
        return out

    return np.apply_along_axis(ss, axis, x, w)


def stmcb(x, u_in=None, q=None, p=None, niter=5, a_in=None):
    """Compute linear model via Steiglitz-McBride iteration.

    [B,A] = stmcb(H,NB,NA) finds the coefficients of the system B(z)/A(z) with
    approximate impulse response H, NA poles and NB zeros.

    [B,A] = stmcb(H,NB,NA,N) uses N iterations.  N defaults to 5.

    [B,A] = stmcb(H,NB,NA,N,Ai) uses the vector Ai as the initial
    guess at the denominator coefficients. If you don't specify Ai,
    STMCB uses [B,Ai] = PRONY(H,0,NA) as the initial conditions.

    [B,A] = STMCB(X,Y,NB,NA,N,Ai) finds the system coefficients B and
    A of the system which, given X as input, has Y as output. N and Ai
    are again optional with default values of N = 5, [B,Ai] = PRONY(Y,0,NA).
    Y and X must be the same length.

    Parameters
    ----------
    x : array
    u_in : array
    q : int
    p : int
    n_iter : int
    a_in : array

    Returns
    -------
    b : array
        Filter coefficients (denominator).
    a : array
        Filter coefficients (numerator).

    Examples
    --------
    Approximate the impulse response of a Butterworth filter with a
    system of lower order:

    >>> [b, a] = butter(6, 0.2)                # Butterworth filter design
    >>> h = filter(b, a, [1, zeros(1,100)])    # Filter data using above filter
    >>> freqz(b, a, 128)                       # Frequency response
    >>> [bb, aa] = stmcb(h, 4, 4)
    >>> plt.plot(freqz(bb, aa, 128))

    References
    ----------
    Authors: Jim McClellan, 2-89 T. Krauss, 4-22-93, new help and options
    Copyright 1988-2004 The MathWorks, Inc.

    """
    if u_in is None:
        if q is None:
            q = 0
        if a_in is None:
            a_in, _ = prony(x, 0, p)

        # make a unit impulse whose length is same as x
        u_in = np.zeros(len(x))
        u_in[0] = 1.
    else:
        if len(u_in) != len(x):
            raise ValueError(
                f"stmcb: u_in and x must be of the same size: {len(u_in)} != {len(x)}")
        if a_in is None:
            q = 0
            _, a_in = prony(x, q, p)

    a = a_in
    N = len(x)
    for _i in range(niter):
        u = lfilter([1], a, x)
        v = lfilter([1], a, u_in)
        C1 = convmtx(u, (p + 1)).T
        C2 = convmtx(v, (q + 1)).T
        T = np.hstack((-C1[0:N, :], C2[0:N, :]))

        # move 1st column to RHS and do least-squares
        # c = T(:,2:p+q+2)\( -T(:,1));
        #
        # If not squared matrix: numpy.linalg.lstsq
        # If squared matrix: numpy.linalg.solve
        T_left = T[:, 1:p + q + 2]
        T_right = -T[:, 0]
        if T.shape[0] != T.shape[1]:
            c, residuals, rank, singular_values = lstsq(
                T_left, T_right)  # lstsq in python returns more stuff
        else:
            c = solve(T_left, T_right)

        # denominator coefficients
        a_left = np.array([1])
        a_right = c[:p]
        a = np.hstack((a_left, a_right))

        # numerator coefficients
        b = c[p:p + q + 1]

    a = a.T
    b = b.T
    return b, a


def prony(h, nb, na):
    """Prony's method for time-domain IIR filter design.

    [B,A] = PRONY(H, NB, NA) finds a filter with numerator order NB,
    denominator order NA, and having the impulse response in vector H. The IIR
    filter coefficients are returned in length NB+1 and NA+1 row vectors B and
    A, ordered in descending powers of Z.  H may be real or complex.

    If the largest order specified is greater than the length of H, H is padded
    with zeros.

    Parameters
    ----------
    h : array
        Impulse response.
    nb : int
        Numerator order.
    na : int
        Denominator order.

    References
    ----------
    T.W. Parks and C.S. Burrus, Digital Filter Design, John Wiley and Sons,
    1987, p226.

    Notes
    -----
    Copyright 1988-2012 The MathWorks, Inc.

    """
    K = len(h) - 1
    M = nb
    N = na
    if K <= max(M, N):
        # zero-pad input if necessary
        K = max(M, N) + 1
        h[K + 1] = 0

    c = h[0]
    if c == 0:
        c = 1  # avoid division by zero

    H = toeplitz(h / c, np.array(np.hstack((1, np.zeros(K)))))

    # K+1 by N+1
    if K > N:
        # Here we are just getting rid of all the columns after N+1
        H = H[:, :N + 1]

    # Partition H matrix
    H1 = H[:M + 1, :]

    # M+1 by N+1
    h1 = H[M + 1:K + 1, 0]

    # K-M by 1
    H2 = H[M + 1:K + 1, 1:N + 1]

    # K-M by N
    a_right = np.linalg.lstsq(-H2, h1, rcond=None)[0]
    a = np.r_[(np.array([1]), a_right)][None, :]
    b = np.dot(np.dot(c, a), H1.T)
    return b, a
