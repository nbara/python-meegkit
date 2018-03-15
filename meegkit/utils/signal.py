"""Signal tools."""
import numpy as np


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
        The dimension of the smoothing window; should be an odd integer.
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


if __name__ is "__main__":
    import matplotlib.pyplot as plt

    x = (np.random.randn(1000, 1) / 2 +
         np.cos(2 * np.pi * 3 * np.linspace(0, 20, 1000))[:, None])

    plt.figure()
    plt.plot(x)
    plt.plot(smooth(x, 2))
    plt.show()
