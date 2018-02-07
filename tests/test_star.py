"""Test STAR functions."""
import numpy as np
from numpy.testing import assert_allclose

from context import meegkit  # noqa
from meegkit.star import star
from meegkit.utils import normcol, demean


def test_star1():
    """Test STAR 1."""
    # N channels,  1 sinusoidal target, N-3 noise sources, temporally local
    # artifacts on each channel.
    n_samples = 1000
    n_chans = 10
    f = 2

    def sim_data(n_samples, n_chans, f, SNR):
        target = np.sin(np.arange(n_samples) / n_samples * 2 * np.pi * f)
        target = target[:, np.newaxis]
        noise = np.random.randn(n_samples, n_chans - 3)

        x0 = (normcol(np.dot(
            noise, np.random.randn(noise.shape[1], n_chans))) +
            SNR * target * np.random.randn(1, n_chans))
        x0 = demean(x0)
        artifact = np.zeros(x0.shape)
        for k in np.arange(n_chans):
            artifact[k * 100 + np.arange(20), k] = 1
        x = x0 + 20 * artifact
        return x, x0

    # Test SNR=1
    x, x0 = sim_data(n_samples, n_chans, f, SNR=np.sqrt(1))
    y, w, _ = star(x, 2)
    assert_allclose(demean(y), x0)  # check that denoised signal ~ x0

    # Test more unfavourable SNR
    x, x0 = sim_data(n_samples, n_chans, f, SNR=np.sqrt(1e-7))
    y, w, _ = star(x, 2)
    assert_allclose(demean(y), x0)  # check that denoised signal ~ x0


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
