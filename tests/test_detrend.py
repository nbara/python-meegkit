"""Test robust detrending."""
import numpy as np

from meegkit.detrend import regress


def test_regress():
    """Test regression."""
    # Simple regression example, no weights
    # fit random walk
    y = np.cumsum(np.random.randn(1000, 1), axis=0)
    x = np.arange(1000)[:, None]
    x = np.hstack([x, x ** 2, x ** 3])
    [b, z] = regress(y, x)

    # Simple regression example, with weights
    y = np.cumsum(np.random.randn(1000, 1), axis=0)
    w = np.random.rand(*y.shape)
    [b, z] = regress(y, x, w)

    # Downweight 1st half of the data
    y = np.cumsum(np.random.randn(1000, 1), axis=0) + 1000
    w = np.ones(y.shape[0])
    w[:500] = 0
    [b, z] = regress(y, x, w)

    # # Multichannel regression
    y = np.cumsum(np.random.randn(1000, 2), axis=0)
    w = np.ones(y.shape[0])
    [b, z] = regress(y, x, w)
    assert z.shape == (1000, 2)
    assert b.shape == (2, 1)

    # Multichannel regression
    y = np.cumsum(np.random.randn(1000, 2), axis=0)
    w = np.ones(y.shape)
    w[:, 1] == .8
    [b, z] = regress(y, x, w)
    assert z.shape == (1000, 2)
    assert b.shape == (2, 1)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
