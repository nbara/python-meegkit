import numpy as np
from scipy import linalg


def nt_cca(x=None, y=None, lags=None, C=None, m=None, thresh=None):
    """CCA implementation.

    [A,B,R]=nt_cca(x,y,lags,C,m,thresh) - canonical correlation

    Returns
    -------
    A, B: transform matrices
    R: r scores

    Parameters
    ----------
    x,y: column matrices
    lags: positive lag means y delayed relative to x
    C: covariance matrix of [x, y]
    m: number of columns of x
    thresh: discard PCs below this

    Notes
    -----
    Usage 1:
    [A,B,R]=nt_cca(x,y); # CCA of x, y

    Usage 2:
    [A,B,R]=nt_cca(x,y,lags); # CCA of x, y for each value of lags.
    A positive lag indicates that y is delayed relative to x.

    Usage 3:
    C=[x,y]'*[x,y]; # covariance
    [A,B,R]=nt_cca([],[],[],C,size(x,2)); # CCA of x,y
    Use the third form to handle multiple files or large data
    (covariance C can be calculated chunk-by-chunk).
    C can be 3-D, which case CCA is derived independently from each page.

    Warning: means of x and y are NOT removed.
    Warning: A, B scaled so that (x*A)^2 and (y*B)^2 are identity matrices
    (differs from canoncorr).

    See Also
    --------
    nt_cov_lags, nt_relshift, nt_cov, nt_pca in NoiseTools.

    """
    if thresh is None:
        thresh = 10.0 ** - 12

    if x is not None:
        if y is None:
            raise AttributeError('!')
        if lags is None:
            lags = np.array([])

        if len(lags) == 1 and lags == 0 and x and x.ndim == 2:
            C = np.dot(np.cat(x, y).T, np.cat(x, y))
            m = x.shape[1]
        else:
            C, __, m = nt_cov_lags(x, y, lags, nargout=3)

        A, B, R = nt_cca([], [], [], C, m, thresh, nargout=3)
        return A, B, R

    # if logical_not(exist('C', 'var')) or isempty(C):
    #     error('!')

    # if logical_not(exist('m', 'var')):
    #     error('!')

    # if C.shape(1) != C.shape(2):
    #     error('!')

    # if logical_not(isempty(x)) or logical_not(isempty(y)) or logical_not(isempty(lags)):
    #     error('!')

    if C.ndim > 3:
        pass
        # error('!')

    if C.ndim == 3:  # covariance is 3D: do a separate CCA for each page

        N = min(m, C.shape[0] - m)
        A = np.zeros(m, N, C.shape(2))
        B = np.zeros(C.shape(0) - m, N, C.shape(2))
        R = np.zeros(N, C.shape(2))

        for k in np.arange(1, C.shape(2)).reshape(-1):
            AA, BB, RR = nt_cca(None, None, None, C[:, :, k], m, nargout=3)
            A[:AA.shape(0), :AA.shape(1), k] = AA
            B[:BB.shape(0), :BB.shape(1), k] = BB
            R[:RR.shape(1), k] = RR

        return A, B, R

    # Calculate CCA given C=[x,y]'*[x,y] and m=size(x,2);
    # -------------------------------------------------------------------------
    # see here for better code :
    # https://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    # sphere x
    Cx = C[:m, :m]
    V, S = linalg.eig(Cx)
    V = np.real(V)
    S = np.real(S)
    E, idx = np.sort(np.diag(S).T, 'descend', nargout=2)
    keep = np.find(E / max(E) > thresh)
    topcs = V[:, idx[keep]]
    E = E[keep]
    EXP = 1 - 10 ** - 12
    E = E ** EXP
    A1 = np.dot(topcs, np.diag(np.sqrt((1.0 / E))))

    # sphere y
    Cy = C[m + 1:, m + 1:]
    V, S = linalg.eig(Cy)
    V = np.real(V)
    S = np.real(S)
    E, idx = np.sort(np.diag(S).T, 'descend', nargout=2)
    keep = np.find(E / max(E) > thresh)
    topcs = V[:, idx[keep]]
    E = E[keep]
    E = E ** EXP
    A2 = np.dot(topcs, np.diag(np.sqrt((1. / E))))

    # apply sphering matrices to C
    AA = np.zeros(A1.shape(0) + A2.shape(0), A1.shape(1) + A2.shape(1))
    AA[:A1.shape(0), :A1.shape(1)] = A1
    AA[A1.shape(0) + 1:, A1.shape(1) + 1:] = A2
    C = np.dot(np.dot(AA.T, C), AA)
    N = min(A1.shape(1), A2.shape(1))

    # PCA
    V, S = linalg.eig(C)

    # [V, S] = eigs(C,N) ; # not faster
    V = np.real(V)
    S = np.real(S)
    E, idx = np.sort(np.diag(S), 'descend', nargout=2)
    topcs = V[:, idx]
    A = np.dot(np.dot(A1, topcs[:A1.shape(1), :N]), np.sqrt(2))
    B = np.dot(np.dot(A2, topcs[A1.shape(1) + 1:, :N]), np.sqrt(2))
    R = E[:N] - 1

    return A, B, R


def whiten(X, fudge=1E-18):
    """Whiten matrix X.

    References
    ----------
    https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca

    """
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = np.dot(X.T, X)

    # eigenvalue decomposition of the covariance matrix
    d, V = linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d + fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


def svd_whiten(X):
    """SVD whitening."""
    U, s, Vt = linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white
