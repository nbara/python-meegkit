import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from meegkit.utils import convmtx, tscov, tsxcov
from meegkit.utils.covariances import nonlinear_eigenspace

rng = np.random.default_rng(10)


def test_nonlinear_eigenspace_reproducible():
    """nonlinear_eigenspace returns identical results across calls."""
    A = rng.standard_normal((6, 6))
    L = A @ A.T + np.eye(6)
    d1, v1 = nonlinear_eigenspace(L, 6)
    d2, v2 = nonlinear_eigenspace(L, 6)
    np.testing.assert_array_equal(np.real(d1), np.real(d2))
    np.testing.assert_array_equal(v1, v2)

def test_tscov():
    """Test time-shift covariance."""
    x = 2 * np.eye(3) + 0.1 * rng.random(3)
    x = x - np.mean(x, 0)

    # Compare 0-lag case with numpy.cov()
    c1, n1 = tscov(x, [0])
    c2 = np.cov(x, bias=False)
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


def test_convmtx():
    """Convmtx comparison with matlab."""
    h = [1, 2, 3, 2, 1]
    X = convmtx(h, 7)
    print(X)

    np.testing.assert_array_equal(
        X,
        np.array([[1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.,  0.],
                  [0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.,  0.],
                  [0.,  0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.,  0.],
                  [0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.,  0.],
                  [0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  3.,  2.,  1.],
                  ])
    )

    print()
    X = convmtx(np.array(h)[None, :], 7)
    print(X)

    np.testing.assert_equal(
        X,
        np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.],
                  [2.,  1.,  0.,  0.,  0.,  0.,  0.],
                  [3.,  2.,  1.,  0.,  0.,  0.,  0.],
                  [2.,  3.,  2.,  1.,  0.,  0.,  0.],
                  [1.,  2.,  3.,  2.,  1.,  0.,  0.],
                  [0.,  1.,  2.,  3.,  2.,  1.,  0.],
                  [0.,  0.,  1.,  2.,  3.,  2.,  1.],
                  [0.,  0.,  0.,  1.,  2.,  3.,  2.],
                  [0.,  0.,  0.,  0.,  1.,  2.,  3.],
                  [0.,  0.,  0.,  0.,  0.,  1.,  2.],
                  [0.,  0.,  0.,  0.,  0.,  0.,  1.],
                  ])
    )


def test_nonlinear_eigenspace_consistent_eigpairs():
    """Returned eigenvalues must match returned eigenvectors."""
    pytest.importorskip("pymanopt")
    from meegkit.utils.covariances import nonlinear_eigenspace

    A = rng.standard_normal((6, 6))
    L = A.T @ A + np.eye(6) * 1e-6

    S, X = nonlinear_eigenspace(L, 6)
    rayleigh = np.real(np.diag(X.T @ L @ X))

    np.testing.assert_allclose(
        np.sort(np.real(S)),
        np.sort(rayleigh),
        rtol=1e-3,
        atol=1e-6,
    )

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
    # test_tscov()
    # test_convmtx()
