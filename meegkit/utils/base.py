"""Math utils."""
from scipy import linalg


def mrdivide(A, B):
    r"""Matrix right-division (A/B).

    Solves the linear system XB = A for X. We can write equivalently:

    1) XB = A
    2) (XB).T = A.T
    3) B.T X.T = A.T

    Therefore A/B amounts to solving B.T X.T = A.T for X.T:

    >> mldivide(B.T, A.T).T

    Parameters
    ----------
    A : ndarray
        Right-hand side matrix.
    B : ndarray
        Coefficient matrix.

    Returns
    -------
    ndarray
        Solution matrix ``X``.

    References
    ----------
    https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    return mldivide(B.T, A.T).T


def mldivide(A, B):
    r"""Matrix left-division (A\B).

    Solves the AX = B for X. In other words, X minimizes norm(A*X - B), the
    length of the vector AX - B:
    - linalg.solve(A, B) if A is square
    - linalg.lstsq(A, B) otherwise

    Parameters
    ----------
    A : ndarray
        Coefficient matrix.
    B : ndarray
        Right-hand side matrix.

    Returns
    -------
    ndarray | None
        Solution matrix, or ``None`` if no stable solution is found.

    References
    ----------
    https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html

    """
    try:
        # Note: we must use overwrite_a=False in order to be able to
        # use the fall-back solution below in case a LinAlgError is raised
        return linalg.solve(A, B, assume_a="pos", overwrite_a=False)
    except linalg.LinAlgError:
        # Singular matrix in solving dual problem. Using least-squares
        # solution instead.
        try:
            return linalg.lstsq(A, B, lapack_driver="gelsy")[0]
        except linalg.LinAlgError:
            print("Solution not stable. Model not updated!")
            return None
