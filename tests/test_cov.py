import numpy as np
from numpy.testing import assert_almost_equal

from context import meegkit  # noqa
from meegkit.utils import tscov, tsxcov


def test_tscov():
    """Test time-shift covariance."""
    x = 2 * np.eye(3) + 0.1 * np.random.rand(3)
    x = x - np.mean(x, 0)

    # Compare 0-lag case with numpy.cov()
    c1, n1 = tscov(x, [0])
    c2 = np.cov(x, bias=True)
    assert_almost_equal(c1 / n1, c2)

    # Compare 0-lag case with numpy.cov()
    x = 2 * np.eye(3)
    c1, n1 = tscov(x, [0, -1])

    assert_almost_equal(c1, np.array([[4, 0, 0, 4, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 4, 0, 0, 4],
                                      [4, 0, 0, 4, 0, 0],
                                      [0, 0, 0, 0, 4, 0],
                                      [0, 0, 4, 0, 0, 4]]))

    c2, n2 = tsxcov(x, x, [0, -1])
    # C3 = nt_tsxcov(x, x, 1:2)
    # C4 = nt_cov_lags(x, x, 1:2)

    # C3 =
    #      0     0     4     0     0     4
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    # C4(:,:,1) =
    #      0     0     0     0     0     0
    #      0     4     0     4     0     0
    #      0     0     4     0     4     0
    #      0     4     0     4     0     0
    #      0     0     4     0     4     0
    #      0     0     0     0     0     0
    # C4(:,:,2) =
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    #      0     0     4     4     0     0
    #      0     0     4     4     0     0
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
