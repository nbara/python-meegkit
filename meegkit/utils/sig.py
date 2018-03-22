"""Signal processing tools."""
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.signal as ss


def smooth(x, window_len, window='square', axis=0):
    """Smooth a signal using a window with requested size along a given axis.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with
    the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output signal.

    Parameters
    ----------
    x : array
        The input signal.
    window_len : int
        The dimension of the smoothing window.
    window : str
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman' flat window will produce a moving average smoothing.
    axis : int
        Axis along which smoothing will be applied (default: 0).

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
        raise ValueError('Input vector needs to be bigger than window size.')
    if window not in ['square', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError('Unknown window type.')
    if window_len == 0:
        raise ValueError('Smoothing kernel must be at least 1 sample wide')
    if window_len == 1:
        return x

    def _smooth1d(x, n):
        if x.ndim != 1:
            raise ValueError('Smooth only accepts 1D arrays')
        if window == 'square':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        a = x[window_len - 1:0:-1]
        b = x[-2:-window_len - 1:-1]
        s = np.r_[a, x, b]
        out = np.convolve(w / w.sum(), s, mode='same')

        return out[len(a):-len(b)]

    if x.ndim > 1:  # apply along given axis
        y = np.apply_along_axis(_smooth1d, axis, x, n=window_len)
    else:
        y = _smooth1d(x, n=np.abs(window_len))

    return y


def erb_bandwidth(fc):
    """Bandwitdh of an Equivalent Rectangular Bandwidth (ERB).

    Parameters
    ----------
    fc : ndarray
        Center frequency, or center frequencies, of the filter.

    Returns
    -------
    ndarray or float
        Equivalent rectangular bandwidth of the filter(s).
    """
    # In Hz, according to Glasberg and Moore (1990)
    return 24.7 + fc / 9.265


def erb2hz(erb):
    """Convert ERB-rate values to the corresponding frequencies in Hz."""
    f = (1. / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)
    return f


def hz2erb(f):
    """Convert frequencies to the corresponding ERB-rates.

    Notes
    -----
    There is a round-off error in the Glasberg & Moore paper, as 1000 / (24.7 *
    4.37) * log(10) = 21.332 and not 21.4 as is stated.
    """
    # erb = 21.332 * np.sign(f) * np.log10(1 + np.abs(f) * 0.00437)
    erb = 9.2645 * np.sign(f) * np.log(1 + np.abs(f) * 0.00437)
    return erb


def erbspace(flow, fhigh, n):
    """Generate n equidistantly spaced points on ERB scale."""
    audlimits = hz2erb([flow, fhigh])
    y = erb2hz(np.linspace(audlimits[0], audlimits[1], n))

    bw = (audlimits[1] - audlimits[0]) / (n - 1)

    # Set the endpoints to be exactly what the user specified, instead of the
    # calculated values
    y[0] = flow
    y[-1] = fhigh

    return y, bw


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
    b, a = ss.butter(N=n, Wn=cutoff * 2. / sfreq, btype='lowpass')
    return ss.lfilter(b, a, x)


def hilbert_envelope(signal):
    """Calculate the Hilbert envelope of a signal.

    Parameters
    ----------
    signal : array
        Signal on which to calculate the hilbert envelope. The calculation
        is done along the last axis (i.e. ``axis=-1``).

    Returns
    -------
    ndarray

    """
    def next_pow_2(x):
        return 1 if x == 0 else 2**(x - 1).bit_length()

    signal = np.asarray(signal)
    N_orig = signal.shape[-1]
    # Next power of 2.
    N = next_pow_2(N_orig)
    y_h = ss.hilbert(signal, N)
    # Return signal with same shape as original
    return np.abs(y_h[..., :N_orig])


class GammatoneFilterbank():
    """Gammatone Filterbank.

    This class computes the filter coefficients for a bank of Gammatone
    filters. These filters were defined by Patterson and Holdworth for
    simulating the cochlea, and originally implemented in [1]_.

    The python was adapted from Alexandre Chabot-Leclerc's pambox, and Jason
    Heeris' gammatone toolbox:
    https://github.com/achabotl/pambox
    https://github.com/detly/gammatone

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the signals to filter.
    cf : array_like
        Center frequencies of the filterbank.
    b : float
        beta of the gammatone filters (default: 1.019).
    order : int
        Order (default: 1).
    q : float
        Q-value of the ERB (default: 9.26449).
    min_bw : float
        Minimum bandwidth of an ERB.

    References
    ----------
    .. [1] Slaney, M. (1993). An efficient implementation of the
       Patterson-Holdsworth auditory filter bank. Apple Computer, Perception
       Group, Tech. Rep, 35(8).

    """

    def __init__(self, sfreq, cf, b=1.019, order=1, q=9.26449, min_bw=24.7):
        self.sfreq = sfreq
        try:
            len(cf)
        except TypeError:
            cf = [cf]
        self.cf = np.asarray(cf)
        self.b = b
        self.erb_order = order
        self.q = q
        self.min_bw = min_bw

    def _get_coefs(self):
        """Compute the filter coefficients for a bank of Gammatone filters.

        These filters were defined by Patterson and Holdworth for simulating
        the cochlea.
        """
        T = 1 / self.sfreq
        cf = self.cf
        q = self.q  # Glasberg and Moore Parameters
        width = 1
        min_bw = 24.7
        order = 1

        erb = width * ((cf / q)**order + min_bw**order)**(1 / order)
        B = 1.019 * 2 * np.pi * erb

        arg = 2 * cf * np.pi * T
        vec = np.exp(2j * arg)

        A0 = T
        A2 = 0
        B0 = 1
        B1 = -2 * np.cos(arg) / np.exp(B * T)
        B2 = np.exp(-2 * B * T)

        rt_pos = np.sqrt(3 + 2**1.5)
        rt_neg = np.sqrt(3 - 2**1.5)

        common = -T * np.exp(-(B * T))

        # TODO: This could be simplified to a matrix calculation involving the
        # constant first term and the alternating rt_pos/rt_neg and +/-1 second
        # terms
        k11 = np.cos(arg) + rt_pos * np.sin(arg)
        k12 = np.cos(arg) - rt_pos * np.sin(arg)
        k13 = np.cos(arg) + rt_neg * np.sin(arg)
        k14 = np.cos(arg) - rt_neg * np.sin(arg)

        A11 = common * k11
        A12 = common * k12
        A13 = common * k13
        A14 = common * k14

        gain_arg = np.exp(1j * arg - B * T)

        gain = np.abs((vec - gain_arg * k11) *
                      (vec - gain_arg * k12) *
                      (vec - gain_arg * k13) *
                      (vec - gain_arg * k14) *
                      (T * np.exp(B * T) /
                       (-1 / np.exp(B * T) + 1 + vec *
                        (1 - np.exp(B * T)))) ** 4)

        allfilts = np.ones_like(cf)

        fcoefs = np.column_stack([
            A0 * allfilts, A11, A12, A13, A14, A2 * allfilts,
            B0 * allfilts, B1, B2,
            gain
        ])

        return fcoefs

    def filter(self, X):
        """Filter X along its last dimension.

        Process an input waveform with a gammatone filter bank. This function
        takes a single sound vector, and returns an array of filter outputs,
        one channel per row.

        Parameters
        ----------
        X : ndarray, shape = (n_chan, n_times)
            Signal to filter.

        Returns
        -------
        ndarray
            Filtered signals with shape ``(M, N)``, where ``M`` is the number
            of channels, and ``N`` is the input signal's nubmer of samples.
        """
        coefs = self._get_coefs()
        output = np.zeros((coefs[:, 9].shape[0], X.shape[0]))

        gain = coefs[:, 9]
        # A0, A11, A2
        As1 = coefs[:, (0, 1, 5)]
        # A0, A12, A2
        As2 = coefs[:, (0, 2, 5)]
        # A0, A13, A2
        As3 = coefs[:, (0, 3, 5)]
        # A0, A14, A2
        As4 = coefs[:, (0, 4, 5)]
        # B0, B1, B2
        Bs = coefs[:, 6:9]

        # Loop over channels
        for idx in range(0, coefs.shape[0]):
            # These seem to be reversed (in the sense of A/B order), but that's
            # what the original code did... Replacing these with polynomial
            # multiplications reduces both accuracy and speed.
            y1 = ss.lfilter(As1[idx], Bs[idx], X)
            y2 = ss.lfilter(As2[idx], Bs[idx], y1)
            y3 = ss.lfilter(As3[idx], Bs[idx], y2)
            y4 = ss.lfilter(As4[idx], Bs[idx], y3)
            output[idx, :] = y4 / gain[idx]

        return output


class AuditoryFilterbank(GammatoneFilterbank):
    """Special case of Gammatone filterbank with preset center frequencies."""

    def __init__(self, sfreq, b=1.019, order=1, q=9.26449, min_bw=24.7):

        cf = np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                         30, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                         5000, 6300, 8000])

        super(AuditoryFilterbank, self).__init__(
            sfreq=sfreq, cf=cf, b=b, order=order, q=q, min_bw=min_bw)


if __name__ is "__main__":
    # import matplotlib.pyplot as plt

    # x = (np.random.randn(1000, 1) / 2 +
    #      np.cos(2 * np.pi * 3 * np.linspace(0, 20, 1000))[:, None])

    # plt.figure()
    # plt.plot(x)
    # plt.plot(smooth(x, 2))
    # plt.show()

    f0 = 63
    erbs, _ = erbspace(f0, 8000, 25)
    print(erbs)
    print(len(erbs))
