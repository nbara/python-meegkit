"""Covariance calculation."""
import numpy as np
import pymanopt
# from numpy import linalg
from pymanopt import Problem
from pymanopt.manifolds import Grassmann
from pymanopt.solvers import TrustRegions
from scipy import linalg

from .base import mldivide
from .matrix import (_check_shifts, _check_weights, multishift, relshift,
                     theshapeof, unsqueeze)


def block_covariance(data, window=128, overlap=0.5, padding=True,
                     estimator='cov'):
    """Compute blockwise covariance.

    Parameters
    ----------
    data : array, shape=(n_channels, n_samples)
        Input data (must be 2D)
    window : int
        Window size.
    overlap : float
        Overlap between successive windows.

    """
    from pyriemann.utils.covariance import _check_est

    assert 0 <= overlap < 1, "overlap must be < 1"
    est = _check_est(estimator)
    X = []
    n_chans, n_samples = data.shape
    if padding:  # pad data with zeros
        pad = np.zeros((n_chans, int(window / 2)))
        data = np.concatenate((pad, data, pad), axis=1)

    jump = int(window * overlap)
    ix = 0
    while (ix + window < n_samples):
        X.append(est(data[:, ix:ix + window]))
        ix = ix + jump

    return np.array(X)


def cov_lags(X, Y, shifts=None):
    """Empirical covariance of the joint array [X, Y] with lags.

    Parameters
    ----------
    X: array, shape=(n_times, n_chans_x[, n_trials])
        Time shifted data.
    Y: array, shape=(n_times, n_chans_y[, n_trials])
        Reference data.
    shifts: array, shape=(n_shifts,)
        Positive lag means X is delayed relative to Y.

    Returns
    -------
    C : array, shape=(n_chans_x + n_chans_y, n_chans_x + n_chans_y, n_shifts)
        Covariance matrix (3D if n_shifts > 1).
    tw : float
        Total weight.
    m : int
        Number of columns in X.

    See Also
    --------
    relshift, tscov, tsxcov

    """
    shifts, n_shifts = _check_shifts(shifts)
    X, Y = unsqueeze(X), unsqueeze(Y)

    n_samples, n_chans, n_trials = theshapeof(X)
    n_samples2, n_chans2, n_trials2 = theshapeof(Y)

    if n_samples != n_samples2:
        raise AttributeError('X and Y must have same n_times')
    if n_trials != n_trials2:
        raise AttributeError('X and Y must have same n_trials')
    if n_samples <= max(shifts):
        raise AttributeError('shifts should be no larger than n_samples')

    n_cov = n_chans + n_chans2  # sum of channels of X and Y
    C = np.zeros((n_cov, n_cov, n_shifts))
    for t in np.arange(n_trials):
        for i, s in enumerate(shifts):
            XX, YY = relshift(X[..., t], ref=Y[..., t], shifts=s)
            XY = np.hstack((XX, YY))
            C[:, :, i] += np.dot(XY.T, XY)

    if n_shifts == 1:
        C = np.squeeze(C, 2)

    tw = n_samples * n_trials

    return C, tw, n_chans


def tsxcov(X, Y, shifts=None, weights=None, assume_centered=True):
    """Calculate cross-covariance of X and time-shifted Y.

    This function calculates, for each pair of columns (Xi, Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj.

    Output is a 2D matrix with dimensions .

    Parameters
    ----------
    X, Y : arrays, shape=(n_times, n_chans[, n_trials])
        Data to cross correlate. X can be 1D, 2D or 3D.
    shifts : array
        Time shifts.
    weights : array
        The weights that are applied to X. 1D (if X is 1D or 2D) or 2D (if X is
        3D).
    assume_centered : bool
        If False, remove the mean of X before computing the covariance
        (default=True).

    Returns
    -------
    C : array, shape=(n_chans_x, n_chans_y * n_shifts)
        Cross-covariance matrix.
    tw : total weight

    """
    n_times, n_chans, n_trials = theshapeof(X)
    n_times2, n_chans2, n_trials2 = theshapeof(Y)
    X = unsqueeze(X)
    Y = unsqueeze(Y)

    weights = _check_weights(weights, X)
    shifts, n_shifts = _check_shifts(shifts)

    if not assume_centered:
        X = X - X.mean(0, keepdims=1)
        Y = Y - Y.mean(0, keepdims=1)

    # Apply weights if any
    if weights.any():
        X = np.einsum('ijk,ilk->ijk', X, weights)  # element-wise mult
        weights = weights[:n_times2, :, :]

    # cross covariance
    # C = np.zeros((n_chans * n_shifts, n_chans2 * n_shifts))
    # for t in np.arange(n_trials):
    #     YY, XX = relshift(Y[..., t], ref=X[..., t], shifts=shifts)
    #     XX = XX.reshape(n_times, n_chans * n_shifts)
    #     YY = YY.reshape(n_times2, n_chans2 * n_shifts)
    #     C += np.dot(XX.T, YY)
    C = np.zeros((n_chans, n_chans2 * n_shifts))
    for t in np.arange(n_trials):
        YY = multishift(Y[..., t], shifts=shifts)
        YY = YY.reshape(n_times2, n_chans2 * n_shifts)
        C += np.dot(X[..., t].T, YY)

    if not weights.any():
        tw = n_trials * n_chans2 * YY.shape[0]
    else:
        weights = weights[:YY.shape[0], ...]
        tw = np.sum(weights.flat)

    return C, tw


def tscov(X, shifts=None, weights=None, assume_centered=True):
    """Time shift covariance.

    This function calculates, for each pair [X[i], X[j]] of columns of X, the
    cross-covariance matrix between the time-shifted versions of X[i].

    Parameters
    ----------
    X : array, shape=(n_times, n_chans[, n_trials])
        Data, can be 1D, 2D or 3D.
    shifts : array
        Array of time shifts.
    weights : array
        Weights, 1D (if X is 1D or 2D) or 2D (if X is 3D). The weights are not
        shifted.
    assume_centered : bool
        If False, remove the mean of X before computing the covariance
        (default=True).

    Returns
    -------
    C : array, shape=(n_chans * n_shifts, n_chans * n_shifts)
        Covariance matrix. This matrix is made up of a (n_times, n_times)
        matrix of submatrices of dimensions (n_shifts, n_shifts).
    tw : array
        Total weight (C/tw is the normalized covariance).

    """
    n_times, n_chans, n_trials = theshapeof(X)
    X = unsqueeze(X)

    weights = _check_weights(weights, X)
    shifts, n_shifts = _check_shifts(shifts)

    if not assume_centered:
        X = X - X.mean(0, keepdims=1)

    if weights.any():  # weights
        X = np.einsum('ijk,ilk->ijk', X, weights)  # element-wise mult
        tw = np.sum(weights[:])
    else:  # no weights
        N = 0
        if len(shifts[shifts < 0]):
            N -= np.min(shifts)
        if len(shifts[shifts >= 0]):
            N += np.max(shifts)
        tw = (n_chans * n_shifts - N) * n_trials

    C = np.zeros((n_chans * n_shifts, n_chans * n_shifts))
    for trial in range(n_trials):
        XX = multishift(X[..., trial], shifts)
        XX = XX.reshape(n_times, n_chans * n_shifts)
        C += np.dot(XX.T, XX)

    return C, tw


def convmtx(V, n):
    """Generate a convolution matrix.

    CONVMTX(V,N) returns the convolution matrix for vector V. If V is a column
    vector and X is a column vector of length N, then CONVMTX(V,N)*X is the
    same as CONV(V,X). If R is a row vector and X is a row vector of length N,
    then X*CONVMTX(R,N) is the same as CONV(R,X).

    Given a vector V of length N, an N+n-1 by n convolution matrix is
    generated of the following form:

    ::

            |  V(0)  0      0     ...      0    |
            |  V(1) V(0)    0     ...      0    |
            |  V(2) V(1)   V(0)   ...      0    |
        X = |   .    .      .              .    |
            |   .    .      .              .    |
            |   .    .      .              .    |
            |  V(N) V(N-1) V(N-2) ...  V(N-n+1) |
            |   0   V(N)   V(N-1) ...  V(N-n+2) |
            |   .    .      .              .    |
            |   .    .      .              .    |
            |   0    0      0     ...    V(N)   |

    That is, V is assumed to be causal, and zero-valued after N.

    Parameters
    ----------
    V : array, shape=(N,) or(N, 1) or (1, N)
        Input vector.
    n : int

    Returns
    -------
    t : array, shape=(N * n - 1, n)

    Examples
    --------
    Generate a simple convolution matrix:

    >>> h = [1, 2, 1]
    >>> convmtx(h,7)
    np.array(
        [[1. 2. 1. 0. 0. 0.]
         [0. 1. 2. 1. 0. 0.]
         [0. 0. 1. 2. 1. 0.]
         [0. 0. 0. 1. 2. 1.]]
    )

    """
    V = np.asarray(V)
    if V.ndim == 1:
        V = V[:, None]
    else:
        assert V.shape[0] == 1 or V.shape[1] == 1

    [nr, nc] = V.shape
    V = V.flatten()

    c = np.hstack((V, np.zeros((n - 1))))
    r = np.zeros(n)
    m = len(c)
    x_left = r[n:0:-1]  # reverse order from n to 2 in original code
    x_right = c.flatten()
    x = np.hstack((x_left, x_right))
    cidx = np.arange(0., (m - 1.) + 1).conj().T
    ridx = np.arange(n, (1.) + (-1.), -1.)

    t = np.zeros([len(cidx), len(ridx)])
    counter_cidx = 0
    for c_val in cidx:
        counter_ridx = 0
        for r_val in ridx:
            t[counter_cidx, counter_ridx] = c_val + r_val
            counter_ridx += 1
        counter_cidx += 1

    # Toeplitz subscripts
    t[:] = x[t.astype(int) - 1]

    if nr > nc:
        t = t.T

    return t


def pca(cov, max_comps=None, thresh=0):
    """PCA from covariance.

    Parameters
    ----------
    cov:  array, shape=(n_chans, n_chans)
        Covariance matrix.
    max_comps : int | None
        Maximum number of components to retain after decomposition. ``None``
        (the default) keeps all suprathreshold components (see ``thresh``).
    thresh : float
        Discard components below this threshold.

    Returns
    -------
    V : array, shape=(max_comps, max_comps)
        Eigenvectors (matrix of PCA components).
    d : array, shape=(max_comps,)
        PCA eigenvalues

    """
    if thresh is not None and (thresh > 1 or thresh < 0):
        raise ValueError('Threshold must be between 0 and 1 (or None).')

    d, V = linalg.eigh(cov)
    d = d.real
    V = V.real

    p0 = d.sum()  # total power

    idx = np.argsort(d)[::-1]  # reverse sort ev order
    d = d[idx]
    V = V[:, idx]

    # Truncate weak components
    if thresh is not None:
        idx = np.where(d / d.max() > thresh)[0]
        d = d[idx]
        V = V[:, idx]

    # Keep a fixed number of components
    if max_comps is None:
        max_comps = V.shape[1]
    else:
        max_comps = np.min((max_comps, V.shape[1]))

    V = V[:, np.arange(max_comps)]
    d = d[np.arange(max_comps)]

    var = 100 * d.sum() / p0
    if var < 99:
        print('[PCA] Explained variance of selected components : {:.2f}%'.
              format(var))

    return V, d


def regcov(Cxy, Cyy, keep=np.array([]), threshold=np.array([])):
    """Compute regression matrix from cross covariance.

    Parameters
    ----------
    Cxy : array
        Cross-covariance matrix between data and regressor.
    Cyy : array
        Covariance matrix of regressor.
    keep : array
        Number of regressor PCs to keep (default=all).
    threshold : float
        Eigenvalue threshold for discarding regressor PCs (default=0).

    Returns
    -------
    R : array
        Matrix to apply to regressor to best model data.

    """
    # PCA of regressor
    [V, d] = pca(Cyy, max_comps=keep, thresh=threshold)

    # cross-covariance between data and regressor PCs
    Cxy = Cxy.T
    R = np.dot(V.T, Cxy)

    # projection matrix from regressor PCs
    R = (R.T * 1 / d).T

    # projection matrix from regressors
    R = V @ R  # np.dot(np.squeeze(V), np.squeeze(R))

    # if R.ndim == 1:
    #     R = R[:, None]

    return R


def nonlinear_eigenspace(L, k, alpha=1):
    """Nonlinear eigenvalue problem: total energy minimization.

    This example is motivated in [1]_ and was adapted from the manopt toolbox
    in Matlab.

    TODO : check this

    Parameters
    ----------
    L : array, shape=(n_channels, n_channels)
        Discrete Laplacian operator: the covariance matrix.
    alpha : float
        Given constant for optimization problem.
    k : int
        Determines how many eigenvalues are returned.

    Returns
    -------
    Xsol : array, shape=(n_channels, n_channels)
        Eigenvectors.
    S0 : array
        Eigenvalues.

    References
    ----------
    .. [1] "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
       Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin, SIAM Journal on Matrix
       Analysis and Applications, 36(2), 752-774, 2015.

    """
    n = L.shape[0]
    assert L.shape[1] == n, 'L must be square.'

    # Grassmann manifold description
    manifold = Grassmann(n, k)
    manifold._dimension = 1  # hack

    # A solver that involves the hessian (check if correct TODO)
    solver = TrustRegions()

    # Cost function evaluation
    @pymanopt.function.Callable
    def cost(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        val = 0.5 * np.trace(X.T @ (L * X)) + \
            (alpha / 4) * (rhoX.T @ mldivide(L, rhoX))
        return val

    # Euclidean gradient evaluation
    @pymanopt.function.Callable
    def egrad(X):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # diag(X*X')
        g = L @ X + alpha * np.diagflat(mldivide(L, rhoX)) @ X
        return g

    # Euclidean Hessian evaluation
    # Note: Manopt automatically converts it to the Riemannian counterpart.
    @pymanopt.function.Callable
    def ehess(X, U):
        rhoX = np.sum(X ** 2, 1, keepdims=True)  # np.diag(X * X')
        rhoXdot = 2 * np.sum(X.dot(U), 1)
        h = L @ U + alpha * np.diagflat(mldivide(L, rhoXdot)) @ X + \
            alpha * np.diagflat(mldivide(L, rhoX)) @ U
        return h

    # Initialization as suggested in above referenced paper.
    # randomly generate starting point for svd
    x = np.random.randn(n, k)
    [U, S, V] = linalg.svd(x, full_matrices=False)
    x = U.dot(V.T)
    S0, U0 = linalg.eig(
        L + alpha * np.diagflat(mldivide(L, np.sum(x**2, 1)))
    )

    # Call manoptsolve to automatically call an appropriate solver.
    # Note: it calls the trust regions solver as we have all the required
    # ingredients, namely, gradient and Hessian, information.
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, ehess=ehess,
                      verbosity=0)
    Xsol = solver.solve(problem, U0)

    return S0, Xsol
