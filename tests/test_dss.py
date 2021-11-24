"""Test DSS functions."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from meegkit import dss
from meegkit.utils import fold, rms, tscov, unfold


def create_data(n_samples=100 * 3, n_chans=30, n_trials=100, noise_dim=20,
                n_bad_chans=1, SNR=.1, show=False):
    """Create synthetic data.

    Parameters
    ----------
    n_samples : int
        [description], by default 100*3
    n_chans : int
        [description], by default 30
    n_trials : int
        [description], by default 100
    noise_dim : int
        Dimensionality of noise, by default 20
    n_bad_chans : int
        [description], by default 1

    Returns
    -------
    data : ndarray, shape=(n_samples, n_chans, n_trials)
    source : ndarray, shape=(n_samples,)
    """
    # source
    source = np.hstack((
        np.zeros((n_samples // 3,)),
        np.sin(2 * np.pi * np.arange(n_samples // 3) / (n_samples / 3)).T,
        np.zeros((n_samples // 3,))))[np.newaxis].T
    s = source * np.random.randn(1, n_chans)  # 300 * 30
    s = s[:, :, np.newaxis]
    s = np.tile(s, (1, 1, 100))

    # set first `n_bad_chans` to zero
    s[:, :n_bad_chans] = 0.

    # noise
    noise = np.dot(
        unfold(np.random.randn(n_samples, noise_dim, n_trials)),
        np.random.randn(noise_dim, n_chans))
    noise = fold(noise, n_samples)

    # mix signal and noise
    data = noise / rms(noise.flatten()) + SNR * s / rms(s.flatten())

    if show:
        f, ax = plt.subplots(3)
        ax[0].plot(source[:, 0], label='source')
        ax[1].plot(noise[:, 1, 0], label='noise')
        ax[2].plot(data[:, 1, 0], label='mixture')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()

    return data, source


@pytest.mark.parametrize('n_bad_chans', [0, -1])
def test_dss0(n_bad_chans):
    """Test dss0.

    Find the linear combinations of multichannel data that
    maximize repeatability over trials. Data are time * channel * trials.

    Uses dss0().

    `n_bad_chans` set the values of the first corresponding number of channels
    to zero.
    """
    n_samples = 300
    data, source = create_data(n_samples=n_samples, n_bad_chans=n_bad_chans)

    # apply DSS to clean them
    c0, _ = tscov(data)
    c1, _ = tscov(np.mean(data, 2))
    [todss, _, pwr0, pwr1] = dss.dss0(c0, c1)
    z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

    best_comp = np.mean(z[:, 0, :], -1)
    scale = np.ptp(best_comp) / np.ptp(source)

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped


def test_dss1(show=False):
    """Test DSS1 (evoked)."""
    n_samples = 300
    data, source = create_data(n_samples=n_samples)

    todss, _, pwr0, pwr1 = dss.dss1(data, weights=None, )
    z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

    best_comp = np.mean(z[:, 0, :], -1)
    scale = np.ptp(best_comp) / np.ptp(source)

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped

    # With weights
    weights = np.zeros(n_samples)
    weights[100:200] = 1  # we placed the signal is in the middle of the trial
    todss, _, pwr0, pwr1 = dss.dss1(data, weights=weights)
    z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

    best_comp = np.mean(z[:, 0, :], -1)
    scale = np.ptp(best_comp) / np.ptp(source)

    if show:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(source, label='source')
        ax2.plot(np.mean(data, 2), label='data')
        ax3.plot(best_comp, label='recovered')
        plt.legend()
        plt.show()

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped


@pytest.mark.parametrize('nkeep', [None, 2])
def test_dss_line(nkeep):
    """Test line noise removal."""
    sr = 200
    fline = 20
    nsamples = 10000
    nchans = 10
    x = np.random.randn(nsamples, nchans)
    artifact = np.sin(np.arange(nsamples) / sr * 2 * np.pi * fline)[:, None]
    artifact[artifact < 0] = 0
    artifact = artifact ** 3
    s = x + 10 * artifact

    def _plot(x):
        f, ax = plt.subplots(1, 2, sharey=True)
        f, Pxx = signal.welch(x, sr, nperseg=1024, axis=0,
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

    # 2D case, n_outputs == 1
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep)
    _plot(out)

    # Test n_outputs > 1
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep, nremove=2)
    # _plot(out)

    # Test n_trials > 1
    x = np.random.randn(nsamples, nchans, 4)
    artifact = np.sin(
        np.arange(nsamples) / sr * 2 * np.pi * fline)[:, None, None]
    artifact[artifact < 0] = 0
    artifact = artifact ** 3
    s = x + 10 * artifact
    out, _ = dss.dss_line(s, fline, sr, nremove=1)


def test_dss_line_iter():
    """Test line noise removal."""

    # data = np.load("data/dss_line_iter_test_data.npy") 
    # # time x channel x trial sf=200 fline=50

    sr = 200
    fline = 20
    nsamples = 10000
    nchans = 10
    x = np.random.randn(nsamples, nchans)
    artifact = np.sin(np.arange(nsamples) / sr * 2 * np.pi * fline)[:, None]
    artifact[artifact < 0] = 0
    artifact = artifact ** 3
    s = x + 10 * artifact

    # def _plot(x):
    #     f, ax = plt.subplots(1, 2, sharey=True)
    #     f, Pxx = signal.welch(x, sr, nperseg=1024, axis=0,
    #                           return_onesided=True)
    #     ax[1].semilogy(f, Pxx)
    #     f, Pxx = signal.welch(s, sr, nperseg=1024, axis=0,
    #                           return_onesided=True)
    #     ax[0].semilogy(f, Pxx)
    #     ax[0].set_xlabel('frequency [Hz]')
    #     ax[1].set_xlabel('frequency [Hz]')
    #     ax[0].set_ylabel('PSD [V**2/Hz]')
    #     ax[0].set_title('before')
    #     ax[1].set_title('after')
    #     plt.show()

    # 2D case, n_outputs == 1
    out, _ = dss.dss_line_iter(s, fline, sr, show=True)
    # _plot(out)

    # # Test n_outputs > 1
    # out, _ = dss.dss_line_iter(s, fline, sr)

    # # Test n_trials > 1
    # x = np.random.randn(nsamples, nchans, 4)
    # artifact = np.sin(
    #     np.arange(nsamples) / sr * 2 * np.pi * fline)[:, None, None]
    # artifact[artifact < 0] = 0
    # artifact = artifact ** 3
    # s = x + 10 * artifact
    # out, _ = dss.dss_line_iter(s, fline, sr)

if __name__ == '__main__':
    pytest.main([__file__])
    # create_data(SNR=5, show=True)
    # test_dss1(True)
    # test_dss_line(None)
    # test_dss_line_iter()
