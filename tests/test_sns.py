"""Test SNS functions."""
from numpy.testing import assert_allclose
from scipy.io import loadmat

from meegkit import sns


def test_sns():
    """Test against NoiseTools."""
    mat = loadmat("./tests/data/snsdata.mat")
    x = mat["x"]
    y_sns = mat["y_sns"]
    r_sns0 = mat["y_sns0"]
    cx = mat["cx"]

    r = sns.sns0(cx, n_neighbors=4)
    assert_allclose(r, r_sns0)  # assert our results match Matlab's

    y, _ = sns.sns(x, n_neighbors=4)
    assert_allclose(y, y_sns)  # assert our results match Matlab's


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
