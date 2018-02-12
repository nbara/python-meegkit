import numpy as np

from .utils import (demean, fold, multishift, normcol, pca, regcov, tscov,
                    tsxcov, unfold)


def tsr(X, R, shifts=None, wX=None, wR=None, keep=None, thresh=None):
    """Time-shift regression.

    The basic idea is to project the signal `X` on a basis formed by the
    orthogonalized time-shifted `R`, and remove the projection. Supposing `R`
    gives a good observation of the noise that contaminates `X`, the noise is
    removed. By allowing time shifts, the algorithm finds the optimal FIR
    filter to apply to `R` so as to compensate for any convolutional mismatch
    between `X` and `R`.

    Parameters
    ----------
    X : array, shape = (n_samples, n_chans[, n_trials])
        Data to denoise.
    R : array, shape = (n_samples, n_chans[, n_trials])
        Reference data.
    shifts : array, shape = (n_shifts,)
        Array of shifts to apply to R (default: [0]).
    wX : array, shape = (n_samples, 1, n_trials)
        Weights to apply to `X`.
    wR : array, shape = (n_samples, 1, n_trials)
        Weights to apply to `R`.
    keep : int | None
        Number of shifted-R PCs to retain (default: all).
    thresh : float
        Ignore shifted-R PCs smaller than thresh (default: 1e-12).

    Returns
    -------
    y : array
        Denoised data.
    idx : array
        Data[idx] is aligned with `y`.
    mean : array
        Channel means (removed by TSR).
    weights : array
        Weights applied by TSR.

    """
    if shifts is None:
        shifts = np.array([0])
    if not wX:
        wX = np.array([])
    if not wR:
        wR = np.array([])
    if not keep:
        keep = np.array([])
    if not thresh:
        thresh = 1e-12

    # adjust to make shifts non-negative
    initial_samples = X.shape[0]

    offset1 = np.max((0, -np.min(shifts)))
    idx = np.arange(offset1, X.shape[0])
    X = X[idx, :, :]

    if wX:
        wX = wX[idx, :, :]

    R = R[:R.shape[0] - offset1, :, :]

    if wR:
        wR = wR[0:-offset1, :, :]

    shifts = shifts + offset1  # shifts are now positive

    # adjust size of X
    offset2 = np.max((0, np.max(shifts)))

    idx = np.arange(X.shape[0]) - offset2
    idx = idx[idx >= 0]
    X = X[idx, :, :]

    if wX:
        wX = wX[idx, :, :]

    n_samples_X, n_chans_X, n_trials_X = X.shape
    n_samples_R,  n_chans_R,  n_trials_R = R.shape

    # consolidate weights into single weight matrix
    weights = np.zeros((n_samples_X, 1, n_trials_R))

    if not wX and not wR:
        weights[np.arange(n_samples_X), :, :] = 1
    elif not wR:
        weights[:, :, :] = wX[:, :, :]
    elif not wX:
        for trial in np.arange(n_trials_X):
            wr = multishift(wR[:, :, trial], shifts).min(1)
            weights[:, :, trial] = wr
    else:
        for trial in np.arange(n_trials_X):
            wr = multishift(wR[:, :, trial], shifts).min(1)
            # wr = (wr, wx[0:wr.shape[0], :, trial]).min() # TODO
            weights[:, :, trial] = wr

    wX = weights
    wR = np.zeros((n_samples_R, 1, n_trials_R))
    wR[idx, :, :] = weights

    # remove weighted means
    X, mean1 = demean(X, wX, return_mean=True)
    R = demean(R, wR)

    # equalize power of R channels, the equalize power of the R PCs
    R = normcol(R, wR)
    R = tspca(R)[0]
    R = normcol(R, wR)

    # covariances and cross-covariance with time-shifted refs
    cref, twcref = tscov(R, shifts, wR)
    cxref, twcxref = tsxcov(X, R, shifts, wX)

    # regression matrix of x on time-shifted refs
    regression = regcov(cxref / twcxref, cref / twcref, keep, thresh)

    # TSPCA: clean x by removing regression on time-shifted refs
    y = np.zeros((n_samples_X, n_chans_X, n_trials_X))
    for trial in np.arange(n_trials_X):
        z = np.dot(np.squeeze(multishift(R[:, :, trial], shifts)), regression)
        y[:, :, trial] = X[np.arange(z.shape[0]), :, trial] - z

    y, mean2 = demean(y, wX, return_mean=True)

    idx = np.arange(offset1, initial_samples - offset2)
    mean_total = mean1 + mean2
    weights = wR

    return y, idx, mean_total, weights


def tspca(X, shifts=None, keep=None, threshold=None, weights=None):
    """Time-shift PCA.

    Parameters
    ----------
    X : array, shape = (n_times, n_chans[, n_trials])
        Data array.
    shifts: array, shape = (n_shifts,)
        Array of shifts to apply.
    keep: int
        Number of components shifted regressor PCs to keep (default: all).
    threshold:
        Discard PCs with eigenvalues below this (default: 1e-6).
    weights:
        Ignore samples with absolute value above this.

    Returns
    -------
    components : array
        Principal components array.
    idx : array
        `X[idx]` maps to principal components.

    """
    if not shifts:
        shifts = np.array([0])
    if not keep:
        keep = np.array([])
    if not threshold:
        threshold = 10 ** -6
    if not weights:
        weights = np.array([])

    n_samples, n_chans, n_trials = X.shape

    # offset of z relative to data
    offset = max(0, -min(shifts, 0))
    shifts += offset
    idx = offset + (np.arange(n_samples) - max([shifts]))

    # remove mean
    X = unfold(X)
    X = demean(X, weights)
    X = fold(X, n_samples)

    # covariance
    if not any(weights):
        c = tscov(X, shifts)[0]
    else:
        if sum(weights) == 0:
            raise ValueError("Weights are all zero.")
        c = tscov(X, shifts, weights)[0]

    # PCA matrix
    topcs, eigenvalues = pca(c)

    # truncate
    if keep:
        topcs = topcs[:, np.arange(keep)]
        eigenvalues = eigenvalues[np.arange(keep)]

    if threshold:
        ii = eigenvalues / eigenvalues[0] > threshold
        topcs = topcs[:, ii]
        eigenvalues = eigenvalues[ii]

    # apply PCA matrix to time-shifted data
    components = np.zeros((idx.size, topcs.shape[1], n_trials))

    for trial in np.arange(n_trials):
        components[:, :, trial] = np.dot(
            np.squeeze(multishift(X[:, :, trial], shifts)),
            np.squeeze(topcs))

    return components, idx
