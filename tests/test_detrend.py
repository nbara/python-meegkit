"""Test robust detrending."""
import numpy as np

from meegkit.detrend import regress, detrend


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


def test_detrend():
    """Test detrending."""
    # basic
    x = np.arange(100)[:, None]
    x = x + np.random.randn(*x.shape)
    y, _, _ = detrend(x, 1)

    assert y.shape == x.shape

    # detrend biased random walk
    x = np.cumsum(np.random.randn(1000, 1) + 0.1)
    y, _, _ = detrend(x, 3)

    assert y.shape == x.shape

    # weights
    trend = np.linspace(0, 100, 1000)[:, None]
    data = 3 * np.random.randn(*trend.shape)
    x = trend + data
    x[:100, :] = 100
    w = np.ones(x.shape)
    w[:100, :] = 0
    y, _, _ = detrend(x, 3, None)
    yy, _, _ = detrend(x, 3, w)

    assert y.shape == x.shape
    assert yy.shape == x.shape

    # assert_almost_equal(yy[100:], data[100:], decimal=1)

if __name__ == '__main__':
    # import pytest
    # pytest.main([__file__])
    test_detrend()
