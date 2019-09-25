"""Test DSS functions."""
import matplotlib.pyplot as plt
import numpy as np
from meegkit import dss
from meegkit.utils import demean, fold, rms, tscov, unfold
from numpy.testing import assert_allclose
from scipy import signal


def test_dss0():
    """Test dss0.

    Find the linear combinations of multichannel data that
    maximize repeatability over trials. Data are time * channel * trials.

    Uses dss0().
    """
    # create synthetic data
    n_samples = 100 * 3
    n_chans = 30
    n_trials = 100
    noise_dim = 20  # dimensionality of noise

    # source
    source = np.hstack((
        np.zeros((n_samples // 3,)),
        np.sin(2 * np.pi * np.arange(n_samples // 3) / (n_samples / 3)).T,
        np.zeros((n_samples // 3,))))[np.newaxis].T
    s = source * np.random.randn(1, n_chans)  # 300 * 30
    s = s[:, :, np.newaxis]
    s = np.tile(s, (1, 1, 100))

    # noise
    noise = np.dot(
        unfold(np.random.randn(n_samples, noise_dim, n_trials)),
        np.random.randn(noise_dim, n_chans))
    noise = fold(noise, n_samples)

    # mix signal and noise
    SNR = 0.1
    data = noise / rms(noise.flatten()) + SNR * s / rms(s.flatten())

    # apply DSS to clean them
    c0, _ = tscov(data)
    c1, _ = tscov(np.mean(data, 2))
    [todss, _, pwr0, pwr1] = dss.dss0(c0, c1)
    z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

    best_comp = np.mean(z[:, 0, :], -1)
    scale = np.ptp(best_comp) / np.ptp(source)

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped


def test_dss_line():
    """Test line noise removal."""
    sr = 200
    nsamples = 10000
    nchans = 10
    x = np.random.randn(nsamples, nchans)
    artifact = np.sin(np.arange(nsamples) / sr * 2 * np.pi * 20)[:, None]
    artifact[artifact < 0] = 0
    artifact = artifact ** 3
    s = x + 10 * artifact

    out, _ = dss.dss_line(s, 20, sr)
    f, ax = plt.subplots(1, 2, sharey=True)
    print(out.shape)
    f, Pxx = signal.welch(out, sr, nperseg=1024, axis=0,
                          return_onesided=True)
    ax[1].semilogy(f, Pxx)
    f, Pxx = signal.welch(s, sr, nperseg=1024, axis=0,
                          return_onesided=True)
    ax[0].semilogy(f, Pxx)
    ax[0].set_xlabel('frequency [Hz]')
    ax[1].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('PSD [V**2/Hz]')
    ax[0].set_title('before')
    ax[1].set_title('after')
    plt.show()

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
    # test_dss_line()
