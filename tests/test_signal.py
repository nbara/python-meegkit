"""Test signal utils."""
import numpy as np
from scipy.signal import butter, freqz, lfilter

from meegkit.utils.sig import stmcb, teager_kaiser

rng = np.random.default_rng(9)

def test_teager_kaiser(show=False):
    """Test Teager-Kaiser Energy."""
    x = 2 * rng.random((1000, 2)) - 1
    x[100, 0] = 5
    x[200, 0] = 5
    x += np.cumsum(rng.standard_normal((1000, 1)) + 0.1, axis=0) / 1000
    for i in range(1, 5):
        print(i)
        y = teager_kaiser(x, M=i, m=1)
        assert y.shape[0] == 1000 - 2 * i

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        # plt.plot((x[1:, 0] - x[1:, 0].mean()) / np.nanstd(x[1:, 0]))
        # plt.plot((y[..., 0] - y[..., 0].mean()) / np.nanstd(y[..., 0]))
        plt.plot(x[1:, 0], label="X")
        plt.plot(y[..., 0], label="Y")
        plt.legend()
        plt.show()


def test_stcmb(show=True):
    """Test stcmb."""
    x = np.arange(100) < 1
    [b, a] = butter(6, 0.2)     # Butterworth filter design
    y = lfilter(b, a, x)        # Filter data using above filter

    w, h = freqz(b, a, 128)     # Frequency response
    [bb, aa] = stmcb(y, u_in=None, q=4, p=4, niter=5)
    ww, hh = freqz(bb, aa, 128)

    if show:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(2, 1)
        ax[0].plot(x, label="step")
        ax[0].plot(y, label="filt")
        ax[1].plot(w, np.abs(h), label="real")
        ax[1].plot(ww, np.abs(hh), label="stcmb")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    np.testing.assert_allclose(h, hh, rtol=2)  # equal to 2%

if __name__ == "__main__":
    test_teager_kaiser()
    test_stcmb()
