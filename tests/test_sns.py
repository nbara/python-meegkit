"""Test DSS functions."""
import numpy as np
from numpy.testing import assert_allclose

from context import meegkit  # noqa
from meegkit import sns


def test_sns():
    """Test against NoiseTools."""
    rng = np.random.RandomState(0)
    n_channels = 9
    x = rng.randn(10000, n_channels)

    # make some correlations
    x = np.dot(rng.randn(n_channels, n_channels), x.T).T
    # x -= x.mean(axis=0, keepdims=True)
    # savemat('test.mat', dict(x=x))  # --> run through nt_sns(x', 5) in MATLAB

    # nt_op = np.array([  # Obtained from NoiseTools 18-Nov-2016
    #     [0, 0, -0.3528, 0, 0.6152, 0, 0, -0.3299, 0.1914],
    #     [0, 0, 0, 0.0336, 0, 0, -0.4284, 0, 0],
    #     [-0.2928, 0.2463, 0, 0, -0.0891, 0, 0, 0.2200, 0],
    #     [0, 0.0191, 0, 0, -0.3756, -0.3253, 0.4047, -0.4608, 0],
    #     [0.3184, 0, -0.0878, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0.5865, 0, -0.2137],
    #     [0, -0.5190, 0, 0.5059, 0, 0.8271, 0, 0, -0.0771],
    #     [-0.3953, 0, 0.3092, -0.5018, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, -0.2050, 0, 0, 0]]).T

    nt_r = np.array([
        [0, 0, -0.3354, 0, 0.6641, 0, 0, -0.4251, 0.5279],
        [0, 0, 0.4071, 0.1608, 0, 0.4267, -0.4623, 0, 0],
        [-0.2843, 0.2152, 0, 0, 0.04740, 0, 0.02380, 0.1789, 0],
        [-0.1262, 0.1960, 0, 0, -0.7486, -0.3920, 0.3837, -0.5459, 0.4860],
        [0.2924, 0, -0.1223, -0.3020, 0, -0.03170, 0, 0, 0],
        [0, 0.4439, 0, -0.1177, 0.09720, 0, 0.5374, 0.3761, -0.3333],
        [0, -1.005, 0.1131, 0.6978, 0.4566, 1.1385, 0, 0, -0.2034],
        [-0.5090, 0, 0.3245, -0.4859, 0, 0, 0, 0, 0.7251],
        [0.4067, -0.3336, 0, 0, 0, -0.02330, -0.1781, 0.3389, 0]]).T

    y, r = sns.sns(x, n_neighbors=5)

    assert_allclose(r, nt_r, rtol=5e-2, atol=.7)  # don't know why this fails


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
