"""Test signal utils."""
import numpy as np
from meegkit.utils.sig import teager_kaiser


def test_teager_kaiser(show=False):
    """Test Teager-Kaiser Energy."""
    x = 2 * np.random.rand(1000, 2) - 1
    x[100, 0] = 5
    x[200, 0] = 5
    x += np.cumsum(np.random.randn(1000, 1) + 0.1, axis=0) / 1000
    for i in range(1, 5):
        print(i)
        y = teager_kaiser(x, M=i, m=1)
        assert y.shape[0] == 1000 - 2 * i

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        # plt.plot((x[1:, 0] - x[1:, 0].mean()) / np.nanstd(x[1:, 0]))
        # plt.plot((y[..., 0] - y[..., 0].mean()) / np.nanstd(y[..., 0]))
        plt.plot(x[1:, 0], label='X')
        plt.plot(y[..., 0], label='Y')
        plt.legend()
        plt.show()
