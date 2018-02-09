import numpy as np
from scipy import linalg
from scipy.signal import filtfilt

from .utils import demean, fold, normcol, pca, theshapeof, tscov, unfold, wpwr


def star(x, thresh=1, closest=[], depth=1, pca_thresh=1e-15, n_smooth=10,
         min_prop=0.3, n_iter=3, verbose=True):
    """Sparse time-artifact removal.

    Parameters
    ----------
    x : array, shape = (n_times, n_chans[, n_trials])
        Data to denoise.
    thresh : float
        Threshold for eccentricity measure (default: 1).
    closest : array, shape = (n_chans, n_neighbours)
        Indices of channels that are closest to each channel (default: all).
    depth : int
        Maximum number of channels to fix at each sample (default: 1).
    pca_thresh : float
        Threshold for discarding weak PCs (default: 1e-15)
    n_smooth : int
        Samples, smoothing to apply to eccentricity (default: 10).
    min_prop : float
        Minimum proportion of artifact free at first iteration (default: 0.3).
    n_iter : int
        Iterations to refine c0 (default: 3).
    verbose : bool
        Verbosity.

    Returns
    -------
    y : array
        Denoised data.
    w : array, shape = (n_times, 1)
        0 for parts that needed fixing, 1 elsewhere.
    ww : array, shape = (n_times, n_chans)
        0 for parts that needed fixing, 1 elsewhere.

    See Also
    --------
    sns

    """
    if thresh is None:
        thresh = 1
    if len(closest) > 0 and closest.shape[0] != x.shape[1]:
        raise ValueError('`closest` should have as many rows as n_chans')

    n_samples, n_chans, n_trials = theshapeof(x)
    x = unfold(x)

    p0, _ = wpwr(x)
    x, intercept = demean(x, return_mean=True)
    norm = np.sqrt(np.mean(x ** 2, axis=0))
    x = normcol(x)
    p00, _ = wpwr(x)

    # NaN and zero channels are set to rand, which effectively excludes them
    idx_nan = np.argwhere(np.all(np.isnan(x), axis=0))
    idx_zero = np.argwhere(np.all(x == 0, axis=0))
    if len(idx_nan) > 0:
        x[:, idx_nan] = np.random.randn(n_samples, np.size(idx_nan))
    if len(idx_zero) > 0:
        x[:, idx_zero] = np.random.randn(n_samples, np.size(idx_zero))

    # initial covariance estimate
    x = demean(x)
    c0, _ = tscov(x)

    # Phase 1
    # -------------------------------------------------------------------------
    # Find time intervals where at least one channel is excentric -> w == 0
    # Compute covariance on artifact-free data.

    iter = n_iter
    while iter > 0:
        w = np.ones((n_samples, 1))
        for ch in np.arange(n_chans):
            neighbours = _closest_neighbours(closest, ch, n_chans)

            # Compute channel data estimated from its neighbours
            y = _project(x[:, neighbours], c0, ch, neighbours)

            # Compute excentricity over time
            d = _eccentricity(x[:, ch][:, None], y, w, n_smooth)
            d = d / thresh

            # Aggregate weights over channels
            # w == 0 : artifactual sample
            # w == 1 : clean time sample
            w = np.min((w, (d < 1)), axis=0)  # w==0 for artifact part

        artifact_free = np.mean(w, axis=0)
        if verbose:
            print('proportion artifact free: {}'.format(artifact_free))

        if iter == n_iter and artifact_free < min_prop:
            thresh = thresh * 1.1
            if verbose:
                print('Warning: increasing threshold to {}'.format(thresh))
            w = np.ones(w.shape)
        else:
            iter = iter - 1

    # restrict covariance estimate to non-artifactual part
    x = demean(x, w)
    c0, _ = tscov(x, None, w)  # TODO check tscov with weights

    # Phase 2
    # -------------------------------------------------------------------------
    # We now know which part contains channel-specific artifacts (w==0 for
    # artifact part), and we have an estimate of the covariance matrix of the
    # artifact-free part.

    # Second eccentricity measure
    d = _eccentricity(x, None, w, n_smooth)

    rank = np.argsort(d, axis=1)[:, ::-1].astype(float)
    rank[np.where(w)[0], :] = np.nan  # exclude parts that are not excentric

    depth = np.min((depth, n_chans - 1))
    ww = np.ones(x.shape)
    for i_depth in np.arange(depth):
        # Fix each channel by projecting on other channels
        i_fixed = n_chans
        n_fixed = 0
        for ch in np.arange(n_chans):
            neighbours = _closest_neighbours(closest, ch, n_chans)

            # samples where this channel is the most excentric
            bad_samples = np.where(ch == rank[:, i_depth])[0]
            if i_depth != 0:  # exclude if not very bad
                bad_samples = np.delete(
                    bad_samples, np.where(d[bad_samples, ch] < thresh)[0])

            n_fixed = n_fixed + np.size(bad_samples)
            if len(bad_samples) == 0:
                i_fixed = i_fixed - 1
                continue

            ww[bad_samples, ch] = 0

            # project this channel on other channels
            y = _project(x[bad_samples, :][:, neighbours], c0, ch, neighbours)
            x[bad_samples, ch] = y.squeeze()  # fix

        if verbose:
            print('depth: {}'.format(i_depth + 1))
            print('fixed channels: {}'.format(i_fixed))
            print('fixed samples: {}'.format(n_fixed))
            print('ratio: {:.2f}'.format(wpwr(x)[0] / p00))

    x = demean(x)
    x = x * norm
    x = x + intercept
    x = fold(x, n_samples)
    w = fold(w, n_samples)
    ww = fold(ww, n_samples)

    # Reset nan and flat channels to zero
    if idx_nan.any():
        x[:, idx_nan] = np.nan
    if idx_zero.any():
        x[:, idx_zero] = 0

    if verbose:
        print('power ratio: {:.2f}'.format(wpwr(x)[0] / p0))

    return x, w, ww


def _closest_neighbours(closest, ch, n_chans):
    """Find indices of closes neighbours to a given channel."""
    if len(closest) > 0:
        neighbours = closest[ch, :]
    else:
        neighbours = np.delete(np.arange(n_chans), ch)

    # in case closest includes channels not in data
    neighbours = np.delete(neighbours, np.where(neighbours > n_chans)[0])

    return neighbours


def _eccentricity(x, y, weights, n_smooth=0):
    """Compute (smoothed) eccentricity of each time point.

    If y is not defined, the eccentricity measure is based on the absolute
    value of the signal.

    If both x and y are defined, the eccentricity is based on distance between
    x (a given channel's data), and y (the projection from all other channels).

    """
    if weights is None:
        weights = np.ones((x.shape[0], 1))

    if y is None:
        e = np.abs(x) / np.nanstd(x[np.where(weights)[0], :], axis=0)
    else:
        eps = np.finfo(float).eps
        # difference from projection, add eps to avoid error on simulated data
        dx = np.abs(y - x) + eps
        e = dx / np.mean(dx[weights > 0], axis=0)  # eccentricity measure

    # Smooth eccentricity slightly over time
    if n_smooth > 0:
        e = filtfilt(np.ones((n_smooth,)) / n_smooth, 1, e, axis=0)

    return e


def _project(X, c0, ch, neighbours, pca_threshold=1e-15):
    """Compute projection of a channel on other channels."""
    # PCA other channels to remove weak dimensions
    c01 = c0[neighbours, :][:, neighbours]
    topcs, eigenvalues = pca(c01)
    idx = np.where(eigenvalues / np.max(eigenvalues) > pca_threshold)[0]
    topcs = topcs[:, idx]

    # X = c0[ch, neighbours].dot(topcs) / (topcs.T.dot(c01).dot(topcs))
    # In Matlab X = A / B performs mrdivide(), and solves for XB=A
    # In python, the closest equivalent is np.dot(A, linalg.pinv(B))
    A = c0[ch, neighbours].dot(topcs)[None, :]
    B = topcs.T.dot(c01).dot(topcs)
    proj = np.dot(A, linalg.pinv(B))
    y = X.dot(topcs.dot(proj.T))  # projection

    return y
