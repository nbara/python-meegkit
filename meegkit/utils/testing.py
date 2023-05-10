"""Synthetic test data."""
import matplotlib.pyplot as plt
import numpy as np

from meegkit.utils import fold, rms, unfold


def create_line_data(n_samples=100 * 3, n_chans=30, n_trials=100, noise_dim=20,
                     n_bad_chans=1, SNR=.1, fline=1, t0=None, show=False):
    """Create synthetic data.

    Parameters
    ----------
    n_samples : int
        Number of samples (default=100*3).
    n_chans : int
        Number of channels (default=30).
    n_trials : int
        Number of trials (default=100).
    noise_dim : int
        Dimensionality of noise (default=20).
    n_bad_chans : int
        Number of bad channels (default=1).
    t0 : int
        Onset sample of artifact.
    fline : float
        Normalized frequency of artifact (freq/samplerate), (default=1).

    Returns
    -------
    data : ndarray, shape=(n_samples, n_chans, n_trials)
    source : ndarray, shape=(n_samples,)
    """
    rng = np.random.RandomState(2022)

    if t0 is None:
        t0 = n_samples // 3
    t1 = n_samples - 2 * t0  # artifact duration

    # create source signal
    source = np.hstack((
        np.zeros(t0),
        np.sin(2 * np.pi * fline * np.arange(t1)),
        np.zeros(t0)))  # noise -> artifact -> noise
    source = source[:, None]

    # mix source in channels
    s = source * rng.randn(1, n_chans)
    s = s[:, :, np.newaxis]
    s = np.tile(s, (1, 1, n_trials))  # create trials

    # set first `n_bad_chans` to zero
    s[:, :n_bad_chans] = 0.

    # noise
    noise = np.dot(
        unfold(rng.randn(n_samples, noise_dim, n_trials)),
        rng.randn(noise_dim, n_chans))
    noise = fold(noise, n_samples)

    # mix signal and noise
    data = noise / rms(noise.flatten()) + SNR * s / rms(s.flatten())

    if show:
        f, ax = plt.subplots(3)
        ax[0].plot(source.mean(-1), label="source")
        ax[1].plot(noise[:, 1].mean(-1), label="noise (avg over trials)")
        ax[2].plot(data[:, 1].mean(-1), label="mixture (avg over trials)")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()

    return data, source
