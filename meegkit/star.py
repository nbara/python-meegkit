"""Sparse time-artefact removal."""
import numpy as np
from scipy.signal import filtfilt

from .utils import (demean, fold, mrdivide, normcol, pca, theshapeof, tscov,
                    unfold, wpwr)


def star(X, thresh=1, closest=[], depth=1, pca_thresh=1e-15, n_smooth=10,
         min_prop=0.5, n_iter=3, verbose=True):
    """Sparse time-artifact removal.

    Parameters
    ----------
    X : array, shape=(n_times, n_chans[, n_trials])
        Data to denoise.
    thresh : float
        Threshold for eccentricity measure (default=1).
    closest : array, shape=(n_chans, n_neighbours)
        Indices of channels that are closest to each channel (default=all).
    depth : int
        Maximum number of channels to fix at each sample (default=1).
    pca_thresh : float
        Threshold for discarding weak PCs (default=1e-15)
    n_smooth : int
        Samples, smoothing to apply to eccentricity (default=10).
    min_prop : float
        Minimum proportion of artifact free at first iteration (default=0.3).
    n_iter : int
        Iterations to refine c0 (default=3).
    verbose : bool | 'debug'
        Verbosity. If 'debug', also draw some diagnostics plots.

    Returns
    -------
    y : array
        Denoised data.
    w : array, shape=(n_times, 1)
        0 for parts that needed fixing, 1 elsewhere.
    ww : array, shape=(n_times, n_chans)
        0 for parts that needed fixing, 1 elsewhere.

    See Also
    --------
    sns

    """
    if thresh is None:
        thresh = 1
    if len(closest) > 0 and closest.shape[0] != X.shape[1]:
        raise ValueError('`closest` should have as many rows as n_chans')

    ndims = X.ndim
    n_samples, n_chans, n_trials = theshapeof(X)
    X = unfold(X)

    p0, _ = wpwr(X)
    X, intercept = demean(X, return_mean=True)
    norm = np.sqrt(np.mean(X ** 2, axis=0))
    X = normcol(X)
    p00, _ = wpwr(X)

    # NaN and zero channels are set to rand, which effectively excludes them
    idx_nan = np.all(np.isnan(X), axis=0)
    idx_zero = np.all(X == 0, axis=0)
    if idx_nan.any():
        X[:, idx_nan] = np.random.randn(X.shape[0], np.sum(idx_nan))
    if idx_zero.any():
        X[:, idx_zero] = np.random.randn(X.shape[0], np.sum(idx_zero))

    # initial covariance estimate
    X = demean(X)
    c0, _ = tscov(X)

    # Phase 1
    # -------------------------------------------------------------------------
    # Find time intervals where at least one channel is eccentric -> w == 0
    # Compute covariance on artifact-free data.

    iter = n_iter
    while iter > 0:
        w = np.ones((X.shape[0],))
        d = np.zeros_like(X)
        for ch in np.arange(n_chans):
            neighbours = _closest_neighbours(closest, ch, n_chans)

            # Compute channel data estimated from its neighbours
            z = _project_channel(X[:, neighbours], c0, ch, neighbours)

            # Compute eccentricity over time
            d[:, ch] = _eccentricity(X[:, ch][:, None], z, w, n_smooth).T
            d[:, ch] = d[:, ch] / thresh

            # Aggregate weights over channels
            # w == 0 : artifactual sample
            # w == 1 : clean time sample
            w = np.min((w, (d[:, ch] < 1)), axis=0)  # w==0 for artifact part

        artifact_free = np.mean(w, axis=0)
        if verbose:
            print('proportion artifact free: {:.2f}'.format(artifact_free))

        if iter == n_iter and artifact_free < min_prop:
            thresh = thresh * 1.1
            if verbose:
                print('Warning: increasing threshold to {:.2f}'.format(thresh))
            w = np.ones(w.shape)
        else:
            iter = iter - 1

    # restrict covariance estimate to non-artifactual part
    X = demean(X, w)
    c0, _ = tscov(X, None, w)

    # Phase 2
    # -------------------------------------------------------------------------
    # We now know which part contains channel-specific artifacts (w==0 for
    # artifact part), and we have an estimate of the covariance matrix of the
    # artifact-free part.

    # Second eccentricity measure
    d = _eccentricity(X, None, w, n_smooth)

    rank = np.argsort(d, axis=1)[:, ::-1].astype(float)
    rank[np.where(w)[0], :] = np.nan  # exclude parts that are not eccentric

    depth = np.min((depth, n_chans - 1))
    ww = np.ones(X.shape)
    y = X.copy()
    for i_depth in np.arange(depth):
        # Fix each channel by projecting on other channels
        i_fixed = n_chans
        n_fixed = 0
        for ch in np.arange(n_chans):
            neighbours = _closest_neighbours(closest, ch, n_chans)

            # find samples where channel `ch` is the most eccentric
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
            z = _project_channel(y[bad_samples, :][:, neighbours], c0, ch,
                                 neighbours)
            y[bad_samples, ch] = z.squeeze()  # fix

        if verbose:
            print('depth: {}'.format(i_depth + 1))
            print('fixed channels: {}'.format(i_fixed))
            print('fixed samples: {}'.format(n_fixed))
            print('ratio: {:.2f}'.format(wpwr(X)[0] / p00))

    y = demean(y)
    y *= norm
    y += intercept

    # Reset nan and flat channels to zero
    if idx_nan.any():
        y[:, idx_nan] = np.nan
    if idx_zero.any():
        y[:, idx_zero] = 0

    if verbose:
        print('power ratio: {:.2f}'.format(wpwr(y)[0] / p0))

    if verbose == 'debug':
        _diagnostics(X * norm + intercept, y, d, thresh)

    if ndims == 3:  # fold back into trials
        y = fold(y, n_samples)
        w = fold(w, n_samples)
        ww = fold(ww, n_samples)

    return y, w, ww


def _closest_neighbours(closest, ch, n_chans):
    """Find indices of closes neighbours to a given channel."""
    if len(closest) > 0:
        neighbours = closest[ch, :]
    else:
        neighbours = np.delete(np.arange(n_chans), ch)

    # in case closest includes channels not in data
    neighbours = np.delete(neighbours, np.where(neighbours > n_chans)[0])

    return neighbours


def _eccentricity(X, y, weights, n_smooth=0):
    """Compute (smoothed) eccentricity of each time point.

    If y is not defined, the eccentricity measure is based on the absolute
    value of the signal.

    If both X and y are defined, the eccentricity is based on distance between
    X (a given channel's data), and y (the projection from all other channels).

    """
    if weights is None:
        weights = np.ones((X.shape[0], 1))

    if y is None:
        e = np.abs(X) / np.nanstd(X[np.where(weights)[0], :], axis=0)
    else:
        eps = np.finfo(float).eps
        # difference from projection, add eps to avoid error on simulated data
        dx = np.abs(y - X) + eps
        e = dx / np.mean(dx[weights > 0], axis=0)  # eccentricity measure

    # Smooth eccentricity slightly over time
    if n_smooth > 0:
        e = filtfilt(np.ones((n_smooth,)) / n_smooth, 1, e, axis=0)

    return e


def _project_channel(X, c0, ch, neighbours, pca_threshold=1e-15):
    """Compute projection of a channel on other channels."""
    # PCA other channels to remove weak dimensions
    c01 = c0[neighbours, :][:, neighbours]
    topcs, eigenvalues = pca(c01)
    idx = np.where(eigenvalues / np.max(eigenvalues) > pca_threshold)[0]
    topcs = topcs[:, idx]

    # X = c0[ch, neighbours].dot(topcs) / (topcs.T.dot(c01).dot(topcs))
    # In Matlab X = A / B performs mrdivide(), and solves for XB=A
    # In python, the best way to approximate this is with linalg.lstsq
    A = c0[ch, neighbours].dot(topcs)[None, :]
    B = topcs.T.dot(c01).dot(topcs)
    proj = mrdivide(A, B)
    y = X.dot(topcs.dot(proj.T))  # projection

    return y


def _diagnostics(X, y, d, thresh):
    """Draw some diagnostic plots."""
    import matplotlib.pyplot as plt

    # ax1.imshow(w, aspect='auto')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    ax1.plot(X, lw=.5)
    ax1.set_title('Signal + Artifacts')
    ax1.set_xticklabels([])

    ax2.plot(demean(y), lw=.5)
    ax2.set_title('Denoised')
    ax2.set_xticklabels([])

    ax3.plot(X - demean(y), lw=.5)
    ax3.set_title('Residual')
    ax3.set_xticklabels([])

    ax4.plot(d, lw=.5, alpha=.3)
    d[d < thresh] = None
    ax4.plot(d, lw=1)
    ax4.axhline(thresh, lw=2, color='k', ls=':')
    ax4.set_ylim([0, thresh + 1])
    ax4.set_title('Eccentricity')
    ax4.set_xlabel('samples')

    f.set_tight_layout(True)
    plt.show()
