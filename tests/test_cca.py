import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from scipy.io import loadmat

from sklearn.cross_decomposition import CCA

from context import meegkit  # noqa
from meegkit.utils import tscov
from meegkit.cca import nt_cca, cca_crossvalidate


def test_cca():
    """Test CCA."""
    # Compare results with Matlab
    # x = np.random.randn(1000, 11)
    # y = np.random.randn(1000, 9)
    # x = demean(x).squeeze()
    # y = demean(y).squeeze()
    mat = loadmat('./tests/data/ccadata.mat')
    x = mat['x']
    y = mat['y']
    A2 = mat['A2']
    B2 = mat['B2']

    A1, B1, R = nt_cca(x, y)  # if mean(A1(:).*A2(:))<0; A2=-A2; end
    X1 = np.dot(x, A1)
    Y1 = np.dot(y, B1)
    C1 = tscov(np.hstack((X1, Y1)))[0]

    # Sklearn CCA
    cca = CCA(n_components=9, scale=False, max_iter=1e6)
    X2, Y2 = cca.fit_transform(x, y)
    # C2 = tscov(np.hstack((X2, Y2)).T)[0]
    # import matplotlib.pyplot as plt
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.imshow(C1)
    # ax2.imshow(C2)
    # plt.show()
    # assert_almost_equal(C1, C2, decimal=4)

    # Compare with matlab
    X2 = np.dot(x, A2)
    Y2 = np.dot(y, B2)
    C2 = tscov(np.hstack((X2, Y2)))[0]

    assert_almost_equal(C1, C2)


def test_canoncorr():
    """Compare with Matlab's canoncorr."""
    x = np.array([[16, 2, 3, 13],
                  [5, 11, 10, 8],
                  [9, 7, 6, 12],
                  [4, 14, 15, 1]])
    y = np.array([[1, 5, 9],
                  [2, 6, 10],
                  [3, 7, 11],
                  [4, 8, 12]])
    [A, B, R] = nt_cca(x, y)
    canoncorr = np.array([[-0.5000, -0.6708],
                          [-0.5000, 0.2236],
                          [-0.5000, -0.2236],
                          [-0.5000, 0.6708]])

    # close enough
    assert_almost_equal(np.cov(np.dot(x, A).T), np.cov(canoncorr.T), decimal=4)


def test_correlated():
    """Test x & y perfectly correlated."""
    x = np.random.randn(1000, 10)
    y = np.random.randn(1000, 10)

    y = x[:, np.random.permutation(10)]  # +0.000001*y;

    [A1, B1, R1] = nt_cca(x, y)

    C = tscov(np.hstack((np.dot(x, A1), np.dot(y, B1))))[0]

    # import matplotlib.pyplot as plt
    # f, ax1 = plt.subplots(1, 1)
    # im = ax1.imshow(C)
    # plt.colorbar(im)
    # plt.show()

    for i in np.arange(C.shape[0]):
        assert_almost_equal(C[i, i], C[i, (i + 10) % 10], decimal=4)
        assert_almost_equal(C[i, i], C[i, (i - 10) % 10], decimal=4)


def test_cca_lags():
    """Test multiple lags."""
    mat = loadmat('./tests/data/ccadata.mat')
    x = mat['x']
    y = mat['y']
    y[:, :3] = x[:, :3]
    lags = np.arange(-10, 11, 1)
    A1, B1, R1 = nt_cca(x, y, lags)

    # import matplotlib.pyplot as plt
    # f, ax1 = plt.subplots(1, 1)
    # ax1.plot(lags, R1.T)
    # plt.show()

    for c in np.arange(3):  # test first 3 components have peaks at 0
        assert_equal(lags[np.argmax(R1[c, :])], 0)


def test_cca_crossvalidate():
    """Test CCA with crossvalidation."""
    # x = np.random.randn(1000, 11)
    # y = np.random.randn(1000, 9)
    # xx = [x, x, x]
    # yy = [x[:, :9], y, y]

    mat = loadmat('./tests/data/ccadata2.mat')
    xx = mat['x']
    yy = mat['y']
    R1 = mat['R']  # no shifts

    # Test with no shifts
    A, B, R = cca_crossvalidate(xx, yy,)

    assert_almost_equal(R, R1, decimal=2)

    # Test with shifts
    A, B, R = cca_crossvalidate(xx, yy, shifts=[1, 2, 3, 4, 5])
    # assert_almost_equal(R, R2, decimal=1)  # won't work as correlations ~0

    # Create data where 1st comps should be uncorrelated, and 2nd and 3rd comps
    # are very correlated
    x = np.random.randn(1000, 10)
    y = np.random.randn(1000, 10)
    xx = [x, x, x]
    yy = [y, x, x]
    A, B, R = cca_crossvalidate(xx, yy)
    assert_almost_equal(R[:, :, 0], 0, decimal=1)
    assert_almost_equal(R[:, :, 1:], 1, decimal=1)

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
