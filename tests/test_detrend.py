"""Test robust detrending."""
import numpy as np

from meegkit.detrend import regress, detrend, reduce_ringing

from scipy.signal import butter, lfilter


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
    assert b.shape == (2, 3)


def test_detrend(show=False):
    """Test detrending."""
    # basic
    x = np.arange(100)[:, None]  # trend
    source = np.random.randn(*x.shape)
    x = x + source
    y, _, _ = detrend(x, 1)

    assert y.shape == x.shape

    # detrend biased random walk
    x = np.cumsum(np.random.randn(1000, 1) + 0.1)
    y, _, _ = detrend(x, 3)

    assert y.shape == x.shape

    # test weights
    trend = np.linspace(0, 100, 1000)[:, None]
    data = 3 * np.random.randn(*trend.shape)
    data[:100, :] = 100
    x = trend + data
    w = np.ones(x.shape)
    w[:100, :] = 0

    # Without weights – detrending fails
    y, _, _ = detrend(x, 3, None, show=show)

    # With weights – detrending works
    yy, _, _ = detrend(x, 3, w, show=show)

    assert y.shape == x.shape
    assert yy.shape == x.shape
    assert np.all(np.abs(yy[100:] - data[100:]) < 1.)

    # detrend higher-dimensional data
    x = np.cumsum(np.random.randn(1000, 16) + 0.1, axis=0)
    y, _, _ = detrend(x, 1, show=False)

    # detrend higher-dimensional data with order 3 polynomial
    x = np.cumsum(np.random.randn(1000, 16) + 0.1, axis=0)
    y, _, _ = detrend(x, 3, basis='polynomials', show=True)

    # detrend with sinusoids
    x = np.random.randn(1000, 2)
    x += 2 * np.sin(2 * np.pi * np.arange(1000) / 200)[:, None]
    y, _, _ = detrend(x, 5, basis='sinusoids', show=True)


def test_ringing():
    """Test reduce_ringing function."""
    x = np.arange(1000) < 1
    [b, a] = butter(6, 0.2)     # Butterworth filter design
    x = lfilter(b, a, x) * 50    # Filter data using above filter
    x = np.roll(x, 500)
    signal = np.random.randn(1000, 2)
    x = x[:, None] + signal
    y = reduce_ringing(x, samples=np.array([500]))

    # np.testing.assert_array_almost_equal(y, signal, 2)

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
    # test_detrend(False)
