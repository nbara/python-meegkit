import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from scipy.io import loadmat

from sklearn.cross_decomposition import CCA

from context import meegkit  # noqa
from meegkit.utils import tscov, multishift
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


def test_cca2():
    """Simulate correlations."""
    # import matplotlib.pyplot as plt

    x = np.random.randn(10000, 20)
    y = np.random.randn(10000, 8)
    y[:, :2] = x[:, :2]
    # perfectly correlated
    y[:, 2:4] = x[:, 2:4] + np.random.randn(10000, 2)
    #  1/2 correlated
    y[:, 4:6] = x[:, 4:6] + np.random.randn(10000, 2) * 3
    #  1/4 correlated
    y[:, 6:8] = np.random.randn(10000, 2)
    # uncorrelated
    [A, B, R] = nt_cca(x, y)

    assert_almost_equal(np.dot(np.dot(x, A).T, np.dot(x, A)), np.eye(8))
    assert_almost_equal(np.dot(np.dot(y, B).T, np.dot(y, B)), np.eye(8))

    # f, ax = plt.subplots(3, 2)
    # ax[0, 0].imshow(A, aspect='auto')
    # ax[0, 0].set_title('A')
    # ax[0, 1].imshow(B, aspect='auto')
    # ax[0, 1].set_title('B')
    # ax[1, 0].plot(R, '.-')
    # ax[1, 0].set_title('R')
    # ax[1, 1].imshow(np.dot(np.dot(x, A).T, np.dot(x, A)), aspect='auto')
    # ax[1, 1].set_title('covariance of x * A')
    # ax[2, 0].imshow(np.dot(np.dot(y, B).T, np.dot(y, B)), aspect='auto')
    # ax[2, 0].set_title('covariance of y * B')
    # f.set_tight_layout(True)
    # plt.show()


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
    A, B, R = cca_crossvalidate(xx, yy)

    assert_almost_equal(R, R1, decimal=2)

    # Create data where 1st comps should be uncorrelated, and 2nd and 3rd comps
    # are very correlated
    x = np.random.randn(1000, 10)
    y = np.random.randn(1000, 10)
    xx = [x, x, x]
    yy = [y, x, x]
    A, B, R = cca_crossvalidate(xx, yy)
    assert_almost_equal(R[:, :, 0], 0, decimal=1)
    assert_almost_equal(R[:, :, 1:], 1, decimal=1)


def test_cca_crossvalidate_shifts():
    """Test CCA crossvalidation with shifts."""
    n_times, n_trials = 10000, 2
    x = np.random.randn(n_times, 20, n_trials)
    y = np.random.randn(n_times, 8, n_trials)
    # perfectly correlated
    y[:, :2, :] = x[:, :2, :]
    # 1/2 correlated
    y[:, 2:4, :] = x[:, 2:4, :] + np.random.randn(n_times, 2, n_trials)
    # 1/4 correlated
    y[:, 4:6, :] = x[:, 4:6, :] + np.random.randn(n_times, 2, n_trials) * 3
    # uncorrelated
    y[:, 6:8, :] = np.random.randn(n_times, 2, n_trials)

    xx = multishift(x, -np.arange(1, 4), reshape=True, solution='valid')
    yy = multishift(y, -np.arange(1, 4), reshape=True, solution='valid')

    # Test with shifts
    A, B, R = cca_crossvalidate(xx, yy, shifts=[-3, -2, -1, 0, 1, 2, 3])

    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(n_trials, 1)
    # for i in range(n_trials):
    #     ax[i].plot(R[:, :, i].T)
    # f.set_tight_layout(True)
    # plt.show()


def test_cca_crossvalidate_shifts2():
    """Test CCA crossvalidation with shifts."""
    mat = loadmat('./tests/data/ccacrossdata.mat')
    xx = mat['xx2']
    yy = mat['yy2']
    R2 = mat['R'][:, ::-1, :]  # shifts go in reverse direction in noisetools

    # Test with shifts
    A, B, R = cca_crossvalidate(xx, yy, shifts=[-3, -2, -1, 0, 1, 2, 3])

    # correlations are ~ those of Matlab
    assert_almost_equal(R, R2, decimal=3)

    # import matplotlib.pyplot as plt
    # n_trials = xx.shape[-1]
    # f, ax = plt.subplots(n_trials, 2)
    # for i in range(n_trials):
    #     ax[i, 0].plot(R[:, :, i].T)
    #     ax[i, 1].plot(R2[:, :, i].T)
    # f.set_tight_layout(True)
    # plt.show()

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
