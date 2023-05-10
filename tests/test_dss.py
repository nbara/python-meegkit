"""Test DSS functions."""
import os
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from meegkit import dss
from meegkit.utils import create_line_data, fold, tscov, unfold

rng = np.random.default_rng(10)


@pytest.mark.parametrize("n_bad_chans", [0, -1])
def test_dss0(n_bad_chans):
    """Test dss0.

    Find the linear combinations of multichannel data that
    maximize repeatability over trials. Data are time * channel * trials.

    Uses dss0().

    `n_bad_chans` set the values of the first corresponding number of channels
    to zero.
    """
    n_samples = 300
    data, source = create_line_data(n_samples=n_samples, n_bad_chans=n_bad_chans)

    # apply DSS to clean them
    c0, _ = tscov(data)
    c1, _ = tscov(np.mean(data, 2))
    [todss, _, pwr0, pwr1] = dss.dss0(c0, c1)
    z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

    best_comp = np.mean(z[:, 0, :], -1)
    scale = np.ptp(best_comp) / np.ptp(source)

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped


def test_dss1(show=True):
    """Test DSS1 (evoked)."""
    n_samples = 300
    data, source = create_line_data(n_samples=n_samples, fline=.01)

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
        ax1.plot(source, label="source")
        ax2.plot(np.mean(data, 2), label="data")
        ax3.plot(best_comp, label="recovered")
        plt.legend()
        plt.show()

    assert_allclose(np.abs(best_comp), np.abs(np.squeeze(source)) * scale,
                    atol=1e-6)  # use abs as DSS component might be flipped


@pytest.mark.parametrize("nkeep", [None, 2])
def test_dss_line(nkeep):
    """Test line noise removal."""
    sr = 200
    fline = 20
    nsamples = 10000
    nchans = 10
    s = create_line_data(n_samples=3 * nsamples, n_chans=nchans,
                         n_trials=1, fline=fline / sr, SNR=2)[0][..., 0]

    def _plot(x):
        f, ax = plt.subplots(1, 2, sharey=True)
        f, Pxx = signal.welch(x, sr, nperseg=1024, axis=0,
                              return_onesided=True)
        ax[1].semilogy(f, Pxx)
        f, Pxx = signal.welch(s, sr, nperseg=1024, axis=0,
                              return_onesided=True)
        ax[0].semilogy(f, Pxx)
        ax[0].set_xlabel("frequency [Hz]")
        ax[1].set_xlabel("frequency [Hz]")
        ax[0].set_ylabel("PSD [V**2/Hz]")
        ax[0].set_title("before")
        ax[1].set_title("after")
        plt.show()

    # 2D case, n_outputs == 1
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep)
    _plot(out)

    # Test blocksize
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep, blocksize=1000)
    _plot(out)

    # Test n_outputs > 1
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep, nremove=2)
    # _plot(out)

    # Test n_trials > 1
    x = rng.standard_normal((nsamples, nchans, 4))
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
    fline = 25
    n_samples = 9000
    n_chans = 10

    # 2D case, n_outputs == 1
    x, _ = create_line_data(n_samples, n_chans=n_chans, n_trials=1,
                            noise_dim=10, SNR=2, fline=fline / sr)
    x = x[..., 0]

    # RuntimeError when max iterations has been reached
    with pytest.raises(RuntimeError):
        out, _ = dss.dss_line_iter(x, fline + 1, sr,
                                   show=False, n_iter_max=2)

    with TemporaryDirectory() as tmpdir:
        out, _ = dss.dss_line_iter(x, fline + .5, sr,
                                   prefix=os.path.join(tmpdir, "dss_iter_"),
                                   show=True)

    def _plot(before, after):
        f, ax = plt.subplots(1, 2, sharey=True)
        f, Pxx = signal.welch(before[:, -1], sr, nperseg=1024, axis=0,
                              return_onesided=True)
        ax[0].semilogy(f, Pxx)
        f, Pxx = signal.welch(after[:, -1], sr, nperseg=1024, axis=0,
                              return_onesided=True)
        ax[1].semilogy(f, Pxx)
        ax[0].set_xlabel("frequency [Hz]")
        ax[1].set_xlabel("frequency [Hz]")
        ax[0].set_ylabel("PSD [V**2/Hz]")
        ax[0].set_title("before")
        ax[1].set_title("after")
        plt.show()

    _plot(x, out)

    # # Test n_trials > 1 TODO
    x, _ = create_line_data(n_samples, n_chans=n_chans, n_trials=2,
                            noise_dim=10, SNR=2, fline=fline / sr)
    out, _ = dss.dss_line_iter(x, fline, sr, show=False)


def profile_dss_line(nkeep):
    """Test line noise removal."""
    import cProfile
    import io
    from pstats import SortKey, Stats

    sr = 200
    fline = 20
    nsamples = 1000000
    nchans = 99
    s = create_line_data(n_samples=3 * nsamples, n_chans=nchans,
                         n_trials=1, fline=fline / sr, SNR=2)[0][..., 0]

    pr = cProfile.Profile()
    pr.enable()
    out, _ = dss.dss_line(s, fline, sr, nkeep=nkeep)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    pytest.main([__file__])
    # create_data(SNR=5, show=True)
    # test_dss1(True)
    # test_dss_line(2)
    # test_dss_line_iter()
    # profile_dss_line(None)
