"""Math utils."""
from scipy.linalg import lstsq, solve


def mrdivide(A, B):
    r"""Matrix right-division (A/B).

    Solves the linear system XB = A for X. We can write equivalently:

    1) XB = A
    2) (XB).T = A.T
    3) B.T X.T = A.T

    Therefore A/B amounts to solving B.T X.T = A.T for X.T:

    >> mldivide(B.T, A.T).T

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    return mldivide(B.T, A.T).T


def mldivide(A, B):
    r"""Matrix left-division (A\B).

    Solves the AX = B for X. In other words, X minimizes norm(A*X - B), the
    length of the vector AX - B:
    - linalg.solve(A, B) if A is square
    - linalg.lstsq(A, B) otherwise

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    if A.shape[0] == A.shape[1]:
        return solve(A, B)
    else:
        return lstsq(A, B)
