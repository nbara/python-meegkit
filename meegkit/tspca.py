"""Time-shift PCA."""
import numpy as np

from .utils import (demean, fold, multishift, normcol, pca, regcov, tscov,
                    tsxcov, unfold, theshapeof, unsqueeze)
from .utils.matrix import _check_shifts, _check_weights


def tspca(X, shifts=None, keep=None, threshold=None, weights=None,
          demean=False):
    """Time-shift PCA.

    Parameters
    ----------
    X : array, shape=(n_times, n_chans[, n_trials])
        Data array.
    shifts: array, shape=(n_shifts,)
        Array of shifts to apply.
    keep: int
        Number of components (shifted regressor PCs) to keep (default=all).
    threshold:
        Discard PCs with eigenvalues below this (default=1e-6).
    weights : array
        Sample weights.
    demean : bool
        If True, Epochs are centered before comuting PCA (default=0).

    Returns
    -------
    comps : array
        Principal components array.
    V : array, shape=(n_chans, n_comps)
        PCA weights.
    idx : array
        `X[idx]` maps to principal comps.

    """
    shifts, n_shifts = _check_shifts(shifts)
    weights = _check_weights(weights, X)

    n_samples, n_chans, n_trials = theshapeof(X)

    # offset of components relative to data
    offset = np.max((0, -np.min(shifts)))
    shifts += offset
    idx = offset + (np.arange(n_samples - np.max(shifts)))

    # remove mean
    if demean:
        X = unfold(X)
        X = demean(X, weights)
        X = fold(X, n_samples)

    # covariance
    C = tscov(X, shifts, weights)[0]

    # PCA matrix
    V, _ = pca(C, max_comps=keep, thresh=threshold)

    # apply PCA matrix to time-shifted data
    comps = np.zeros((np.size(idx), V.shape[1], n_trials))

    for t in np.arange(n_trials):
        comps[:, :, t] = np.dot(
            np.squeeze(multishift(X[:, :, t], shifts)),
            np.squeeze(V))

    return comps, V, idx


def tsr(X, R, shifts=None, wX=None, wR=None, keep=None, thresh=1e-12):
    """Time-shift regression.

    The basic idea is to project the signal `X` on a basis formed by the
    orthogonalized time-shifted `R`, and remove the projection. Supposing `R`
    gives a good observation of the noise that contaminates `X`, the noise is
    removed. By allowing time shifts, the algorithm finds the optimal FIR
    filter to apply to `R` so as to compensate for any convolutional mismatch
    between `X` and `R`.

    Parameters
    ----------
    X : array, shape=(n_samples, n_chans[, n_trials])
        Data to denoise.
    R : array, shape=(n_samples, n_comps[, n_trials])
        Reference data.
    shifts : array, shape=(n_shifts,)
        Array of shifts to apply to R (default=[0]).
    wX : array, shape=(n_samples, 1, n_trials)
        Weights to apply to `X`.
    wR : array, shape=(n_samples, 1, n_trials)
        Weights to apply to `R`.
    keep : int | None
        Number of shifted-R PCs to retain (default=all).
    thresh : float
        Ignore shifted-R PCs smaller than thresh (default=1e-12).

    Returns
    -------
    y : array
        Denoised data.
    idx : array
        X[idx] is aligned with `y`.
    mean : array
        Channel means (removed by TSR).
    weights : array
        Weights applied by TSR.

    """
    ndims = X.ndim
    X = unsqueeze(X)
    R = unsqueeze(R)
    shifts, n_shifts = _check_shifts(shifts)
    wX = _check_weights(wX, X)
    wR = _check_weights(wR, R)

    # adjust to make shifts non-negative
    initial_samples = X.shape[0]

    offset1 = np.max((0, -np.min(shifts)))
    idx = np.arange(offset1, X.shape[0])
    # X = X[idx, ...]
    # if len(wX) > 0:
    #     wX = wX[idx, ...]
    # if len(wR) > 0:
    #     wR = wR[:-offset1, ...]

    shifts = shifts + offset1  # shifts are now positive

    # adjust size of X
    offset2 = np.max((0, np.max(shifts)))
    # idx = np.arange(X.shape[0]) - offset2
    # idx = idx[idx >= 0]
    # X = X[idx, ...]
    # if len(wX) > 0:
    #     wX = wX[idx, ...]

    n_samples_X, n_chans_X, n_trials_X = theshapeof(X)
    n_samples_R, n_chans_R, n_trials_R = theshapeof(R)

    # consolidate weights into single weight matrix
    weights = np.zeros((n_samples_X, 1, n_trials_R))
    if len(wX) == 0 and len(wR) == 0:
        weights[:] = 1
    elif not wR:
        weights = wX
    elif not wX:
        for t in np.arange(n_trials_X):
            wr = multishift(wR[..., t], shifts, reshape=True).min(axis=1)
            weights[..., t] = wr
    else:
        for t in np.arange(n_trials_X):
            wr = multishift(wR[..., t], shifts, reshape=True).min(axis=1)
            wr = np.amin((wr, wX[:wr.shape[0], :, t]), axis=0)
            weights[..., t] = wr

    wX = weights
    wR = weights

    # remove weighted means
    X, mean1 = demean(X, wX, return_mean=True)
    R = demean(R, wR)

    # equalize power of R channels, the equalize power of the R PCs
    # if R.shape[1] > 1:
    R = normcol(R, wR)
    C, _ = tscov(R)
    V, _ = pca(C, thresh=1e-6)
    z = np.zeros((n_samples_X, V.shape[1], n_trials_R))
    for t in range(n_trials_R):
        z[..., t] = R[..., t] @ V
    R = normcol(z, wR)

    # covariances and cross-covariance with time-shifted refs
    Cr, twcr = tscov(R, shifts, wR)
    Cxr, twcxr = tsxcov(X, R, shifts, wX)

    # regression matrix of x on time-shifted refs
    regression = regcov(Cxr / twcxr, Cr / twcr, keep, thresh)

    # TSPCA: clean x by removing regression on time-shifted refs
    y = np.zeros((n_samples_X, n_chans_X, n_trials_X))
    for t in np.arange(n_trials_X):
        r = multishift(R[..., t], shifts, reshape=True)
        z = r @ regression
        y[..., t] = X[:z.shape[0], :, t] - z

    y, mean2 = demean(y, wX, return_mean=True)

    idx = np.arange(offset1, initial_samples - offset2)
    mean_total = mean1 + mean2
    weights = wR

    if ndims < 3:
        y = y.squeeze(2)
        weights = weights.squeeze(2)

    return y, idx, mean_total, weights
