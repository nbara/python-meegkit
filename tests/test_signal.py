"""Test signal utils."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz, hilbert, lfilter

from meegkit.phase import ECHT
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



def test_echt(show=True):
    """Test Endpoint-corrected Hilbert transform (ECHT) phase estimation."""
    rng = np.random.default_rng(38872)

    # Build data
    # -------------------------------------------------------------------------
    # First, we generate a multi-component signal with amplitude and phase
    # modulations, as described in the paper [1]_.
    f0 = 2
    filt_BW = f0 / 2
    N = 500
    sfreq = 200
    time = np.linspace(0, N / sfreq, N)
    X = np.cos(2 * np.pi * f0 * time - np.pi / 4)
    phase_true = np.angle(hilbert(X))
    X += rng.normal(0, 0.5, N)  # Add noise

    # Compute phase and amplitude
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We compute the Hilbert phase, as well as the phase obtained with the ECHT
    # filter.
    # phase_hilbert = np.angle(hilbert(X))  # Hilbert phase

    # Compute ECHT-filtered signal
    l_freq = f0 - filt_BW / 2
    h_freq = f0 + filt_BW / 2
    echt = ECHT(l_freq, h_freq, sfreq)

    Xf = echt.fit_transform(X)
    phase_echt = np.angle(Xf).squeeze()
    # phase_true = np.roll(phase_true, 1)
    if show:
        fig, ax = plt.subplots(3, 1, figsize=(8, 5))
        ax[0].plot(time, X)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_title("Test signal")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(time, phase_true, label="True phase", ls=":")
        ax[1].plot(time, phase_echt, label="ECHT phase", lw=.5, alpha=.8)
        ax[1].set_title("Phase")
        ax[1].set_ylabel("Amplitude")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend(loc="upper right", fontsize="small")

        ax[2].plot(time, np.unwrap(phase_true - phase_echt.squeeze()),
                   label="Phase error")
        ax[2].set_title("Phase error")
        ax[2].set_ylabel("Error")

        plt.tight_layout()
        plt.show()

    mae =  (np.abs(np.unwrap(phase_true - phase_echt.squeeze())) > np.pi / 6).sum() / N
    assert mae < 0.1, mae



if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
    # test_teager_kaiser()
    # test_stcmb()
    # test_echt()