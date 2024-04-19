"""Test bicoherence functions."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.fftpack import next_fast_len

from meegkit.utils.coherence import (
    cross_coherence,
    plot_polycoherence,
    plot_polycoherence_1d,
    plot_signal,
    polycoherence_0d,
    polycoherence_1d,
    polycoherence_1d_sum,
    polycoherence_2d,
)


@pytest.mark.parametrize("norm", [None, 2])
def test_coherence(norm, show=True):
    rng = np.random.default_rng(54326)

    # create signal with 5Hz and 8Hz components and noise, as well as a
    # 5Hz-8Hz interaction
    N = 10001
    kw = dict(nperseg=N // 4, noverlap=N // 20, nfft=next_fast_len(N // 2))
    t = np.linspace(0, 100, N)
    fs = 1 / (t[1] - t[0])
    s1 = np.cos(2 * np.pi * 5 * t + 0.3)  # 5Hz
    s2 = 3 * np.cos(2 * np.pi * 8 * t + 0.5)  # 8Hz
    noise = 5 * rng.normal(0, 1, N)
    signal = s1 + s2 + noise + 0.5 * s1 * s2

    # bicoherence
    # ---------------------------------------------r----------------------------

    # 0D local bicoherence (fixed frequencies)
    p5_8 = polycoherence_0d(signal, fs, [5, 8], **kw)
    p5_6 = polycoherence_0d(signal, fs, [5, 6], **kw)
    p5_6_8 = polycoherence_0d(signal, fs, [5, 6, 8], **kw)
    if norm is not None:
        print(f"bicoherence for f1=5Hz, f2=8Hz: {p5_8:.2f}")
        print(f"bicoherence for f1=5Hz, f2=6Hz: {p5_6:.2f}")
        print(f"bicoherence for f1=5Hz, f2=6Hz, f3=7Hz: {p5_6_8:.2f}")
        assert p5_8 > 0.85 > p5_6 > p5_6_8  # 5Hz and 7Hz are coherent, 5Hz and 6Hz not

    # 1D bicoherence (fixed f2)
    freqs, coh1d = polycoherence_1d(signal, fs, [5], **kw)

    # 1D bicoherence with sum (fixed f1+f2)
    freqs1dsum, coh1dsum = polycoherence_1d_sum(signal, fs, 13, **kw)

    # 2D bicoherence, span all frequencies
    # assert peaks at intersection of 5Hz and 8Hz, and 8-5=3Hz
    freqs1, freqs2, coh2d = polycoherence_2d(signal, fs, **kw)

    if norm is not None:
        assert np.max(coh2d) > 0.85
        assert np.abs(coh2d[freqs1 == 5, freqs2 == 8]) > 0.85
        assert np.abs(coh2d[freqs1 == 5, freqs2 == 3]) > 0.85


    if show:
        # Plot signal
        plot_signal(t, signal)
        plt.suptitle("signal and spectrum for bicoherence tests")

        # Plot bicoherence
        plot_polycoherence(freqs1, freqs2, coh2d)
        plt.suptitle("bicoherence")

        # Plot bicoherence 1D
        plot_polycoherence_1d(freqs, coh1d)
        plt.suptitle("bicoherence for f2=5Hz (column, expected 3Hz, 8Hz)")

        # Plot bicoherence for f1+f2=13Hz
        plot_polycoherence_1d(freqs1dsum, coh1dsum)
        plt.suptitle("bicoherence for f1+f2=13Hz (diagonal, expected 5Hz, 8Hz)")
        plt.show()


    # bicoherence with synthetic signal
    # -------------------------------------------------------------------------
    s3 = 4 * np.cos(2 * np.pi * 1 * t + 0.1)
    s5 = 0.4 * np.cos(2 * np.pi * 0.2 * t + 1)
    signal = s2 + s3 + s5 - 0.5 * s2 * s3 * s5 + noise

    synthetic = ((0.2, 10, 1), )
    p02_1_8 = polycoherence_0d(signal, fs, [0.2, 1, 8], synthetic=None,
                               norm=norm, **kw)
    p02_1_8s = polycoherence_0d(signal, fs, [0.2, 1, 8], synthetic=synthetic,
                               norm=norm, **kw)
    if norm is not None:
        print(f"coherence for f1=0.02Hz, f2=1Hz, f3=7Hz: {p02_1_8:.2f}")
        print(f"coherence for f1=0.02Hz (synthetic), f2=1Hz, f3=7Hz: {p02_1_8s:.2f}")
        assert p02_1_8s > 0.9 > p02_1_8

    result = polycoherence_2d(signal, fs, [0.02], synthetic=synthetic, norm=norm, **kw)

    if show:
        plot_signal(t, signal)
        plt.suptitle("signal and spectrum for tricoherence with synthetic signals")
        plt.tight_layout()

        plot_polycoherence(*result)
        plt.suptitle("tricoherence with f3=0.2Hz (synthetic)")
        plt.show()

    # cross-coherence
    # -------------------------------------------------------------------------
    s1 = np.cos(2 * np.pi * 5 * t + 0.3)  # 5Hz
    s2 = 3 * np.cos(2 * np.pi * 8 * t + 0.5)  # 8Hz
    signal1 = s1 + 4 * rng.normal(0, 1, N)
    signal2 = s2 + s1 + 5 * rng.normal(0, 1, N)
    f, Cxy = cross_coherence(signal1, signal2, fs, norm=None, **kw)

    if show:
        plot_polycoherence(f, f, Cxy)
        plt.suptitle("cross-coherence between 5Hz and 5+8Hz signals")
        plt.tight_layout()
        plt.show()

@pytest.mark.parametrize("shape", [(1001,), (3, 1001), (17, 3, 1001)])
def test_coherence_shapes(shape):
    """Test coherence functions with 1D, 2D or 3D input data."""
    rng = np.random.default_rng(54326)

    # create signal with 5Hz and 8Hz components and noise, as well as a
    # 5Hz-8Hz interaction
    N = shape[-1]
    t = np.linspace(0, 100, N)
    fs = 1 / (t[1] - t[0])
    s1 = np.cos(2 * np.pi * 5 * t + 0.3)  # 5Hz
    s2 = 3 * np.cos(2 * np.pi * 8 * t + 0.5)  # 8Hz
    noise = 5 * rng.normal(0, 1, shape)
    signal = np.broadcast_to(s1 + s2, shape) + noise

    assert signal.shape == shape

    B = polycoherence_0d(signal, fs, [5, 8])
    assert B.shape == shape[:-1]
    f, B = polycoherence_1d(signal, fs, [8])
    assert B.shape == shape[:-1] + f.shape
    f, B = polycoherence_1d_sum(signal, fs, 13)
    assert B.shape == shape[:-1] + f.shape
    f1, f2, B = polycoherence_2d(signal, fs)
    assert B.shape == shape[:-1] + f1.shape + f2.shape


if __name__ == "__main__":
    pytest.main([__file__])
    # test_coherence(2, False)