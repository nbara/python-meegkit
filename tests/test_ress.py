"""Test RESS."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal as ss
from scipy.linalg import pinv

from meegkit import ress
from meegkit.utils import fold, matmul3d, snr_spectrum, unfold

rng = np.random.default_rng(9)


def create_data(n_times, n_chans=10, n_trials=20, freq=12, sfreq=250,
                noise_dim=8, SNR=.8, t0=100, show=False):
    """Create synthetic data.
    Returns
    -------
    noisy_data: array, shape=(n_times, n_channels, n_trials)
        Simulated data with oscillatory component strting at t0.
    """
    # source
    source = np.sin(2 * np.pi * freq * np.arange(n_times - t0) / sfreq)[None].T
    s = source * rng.standard_normal((1, n_chans))
    s = s[:, :, np.newaxis]
    s = np.tile(s, (1, 1, n_trials))
    signal = np.zeros((n_times, n_chans, n_trials))
    signal[t0:, :, :] = s

    # noise
    noise = np.dot(
        unfold(rng.standard_normal((n_times, noise_dim, n_trials))),
        rng.standard_normal((noise_dim, n_chans)))
    noise = fold(noise, n_times)

    # mix signal and noise
    signal = SNR * signal /  np.sqrt(np.mean(signal ** 2))
    noise = noise / np.sqrt(np.mean(noise ** 2))
    noisy_data = signal + noise

    if show:
        f, ax = plt.subplots(3)
        ax[0].plot(signal[:, 0, 0], label="source")
        ax[1].plot(noise[:, 1, 0], label="noise")
        ax[2].plot(noisy_data[:, 1, 0], label="mixture")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()

    return noisy_data, signal


@pytest.mark.parametrize("target", [12, 15, 20])
@pytest.mark.parametrize("n_trials", [16])
@pytest.mark.parametrize("peak_width", [.5, 1])
@pytest.mark.parametrize("neig_width", [1])
@pytest.mark.parametrize("neig_freq", [1])
def test_ress(target, n_trials, peak_width, neig_width, neig_freq, show=False):
    """Test RESS."""
    sfreq = 250
    n_keep = 1
    n_chans = 10
    n_times = 1000
    data, source = create_data(n_times=n_times, n_trials=n_trials,
                               n_chans=n_chans, freq=target, sfreq=sfreq,
                               show=False)
    r = ress.RESS(sfreq=sfreq, peak_freq=target, neig_freq=neig_freq,
        peak_width=peak_width, neig_width=neig_width, n_keep=n_keep,
        compute_unmixing=True)
    out = r.fit_transform(data)

    nfft = 500
    bins, psd = ss.welch(np.squeeze(out), sfreq, window="boxcar",
                         nperseg=nfft / (peak_width * 2),
                         noverlap=0, axis=0, average="mean")
    # psd = np.abs(np.fft.fft(out, nfft, axis=0))
    # psd = psd[0:psd.shape[0] // 2 + 1]
    # bins = np.linspace(0, sfreq // 2, psd.shape[0])
    # print(psd.shape)
    # print(bins[:10])

    psd = psd.mean(axis=-1, keepdims=True)  # average over trials
    snr = snr_spectrum(psd + psd.max() / 20, bins, skipbins=1, n_avg=2)
    # snr = snr.mean(1)
    if show:
        f, ax = plt.subplots(2)
        ax[0].plot(bins, snr, ":o")
        ax[0].axhline(1, ls=":", c="grey", zorder=0)
        ax[0].axvline(target, ls=":", c="grey", zorder=0)
        ax[0].set_ylabel("SNR (a.u.)")
        ax[0].set_xlabel("Frequency (Hz)")
        ax[0].set_xlim([0, 40])
        ax[0].set_ylim([0, 10])
        ax[1].plot(bins, psd)
        ax[1].axvline(target, ls=":", c="grey", zorder=0)
        ax[1].set_ylabel("PSD")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 40])
        # plt.show()

    assert snr[bins == target] > 10
    assert (snr[(bins <= target - 2) | (bins >= target + 2)] < 2).all()

    # test multiple components

    out = r.transform(data)
    toress = r.to_ress
    fromress = r.from_ress

    proj = matmul3d(out, fromress)
    assert proj.shape == (n_times, n_chans, n_trials)

    if show:
        f, ax = plt.subplots(data.shape[1], 2, sharey="col")
        for c in range(data.shape[1]):
            ax[c, 0].plot(data[:, c].mean(-1), lw=.5, label="data")
            ax[c, 1].plot(proj[:, c].mean(-1), lw=.5, label="projection")
            if c < data.shape[1]:
                ax[c, 0].set_xticks([])
                ax[c, 1].set_xticks([])

        ax[0, 0].set_title("Before")
        ax[0, 1].set_title("After")
        plt.legend()

    # 2 comps
    _ = ress.RESS(
        sfreq=sfreq, peak_freq=target, n_keep=2
    ).fit_transform(data)

    # All comps
    r = ress.RESS(sfreq=sfreq, peak_freq=target, neig_freq=neig_freq,
        peak_width=peak_width, neig_width=neig_width, n_keep=-1, compute_unmixing=True)
    out = r.fit_transform(data)
    toress = r.to_ress
    fromress = r.from_ress

    if show:
        # Inspect mixing/unmixing matrices
        combined_data = np.array([toress, fromress, pinv(toress)])
        _max = np.amax(combined_data)

        f, ax = plt.subplots(3)
        ax[0].imshow(toress, label="toRESS")
        ax[0].set_title("toRESS")
        ax[1].imshow(fromress, label="fromRESS", vmin=-_max, vmax=_max)
        ax[1].set_title("fromRESS")
        ax[2].imshow(pinv(toress), vmin=-_max, vmax=_max)
        ax[2].set_title("toRESS$^{-1}$")
        plt.tight_layout()
        plt.show()

    print(np.sum(np.abs(pinv(toress) - fromress) >= .1))


if __name__ == "__main__":
    import pytest
    # pytest.main([__file__])
    test_ress(20, 16, 1, 1, 1, show=True)
