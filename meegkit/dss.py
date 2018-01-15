import numpy as np
from scipy import linalg
from .utils import demean, tscov, mean_over_trials, pca, theshapeof


def dss1(data, weights=None, keep1=None, keep2=1e-12):
    """DSS to maximise repeatability across trials.

    Evoked-biased DSS desnoising.

    Parameters
    ----------
    data: array, shape = (n_samples, n_chans, n_trials)
        Data to denoise.
    weights: array
        Weights.
    keep1: int
        Number of PCs to retain in function:`dss0` (default: all).
    keep2: float
        Ignore PCs smaller than keep2 in function:`dss0` (default: 10^-12).

    Returns
    -------
    todss: array, shape = (n_dss_components, n_chans)
        Ddenoising matrix to convert data to normalized DSS components.
    pwr0: array
        Power per component (raw).
    pwr1: array
        Power per component (averaged).

    Notes
    -----
    The data mean is NOT removed prior to processing.

    """
    n_samples, n_chans, n_trials = theshapeof(data)
    data, data_mean = demean(data, weights)  # remove weighted mean

    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(data, weights)
    print("xx.shape", xx.shape)
    ww = ww.min(1)

    # covariance of raw and biased data
    c0, nc0 = tscov(data, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c1 = c1 / n_trials

    todss, fromdss, ratio, pwr = dss0(c0, c1, keep1, keep2)

    return todss, fromdss, ratio, pwr


def dss0(c0, c1, keep1=None, keep2=1e-9):
    """DSS base function.

    This function allows specifying arbitrary bias functions (as compared to
    the function:`dss1`, which forces the bias to be the mean over trials).

    Parameters
    ----------
    c0: array, shape = (n_chans, n_chans)
        Baseline covariance.
    c1: array, shape = (n_chans, n_chans)
        Biased covariance.
    keep1: int
        Number of PCs to retain (default: all).
    keep2: float
        Ignore PCs smaller than keep2 (default: 10.^-9).

    Returns
    -------
    todss: array, shape = (n_dss_components, n_chans)
        Matrix to convert data to normalized DSS components.
    pwr0: array
        Power per component (baseline).
    pwr1: array
        Power per component (biased).

    """
    if c0 is None or c1 is None:
        raise AttributeError('dss0 needs at least two arguments')
    if c0.shape != c1.shape:
        raise AttributeError('c0 and c1 should have same size')
    if c0.shape[0] != c0.shape[1]:
        raise AttributeError('c0 should be square')
    if np.any(np.isnan(c0)) or np.any(np.isinf(c0)):
        raise ValueError('NaN or INF in c0')
    if np.any(np.isnan(c1)) or np.any(np.isinf(c1)):
        raise ValueError('NaN or INF in c1')

    # derive PCA and whitening matrix from unbiased covariance
    eigvec0, eigval0 = pca(c0, max_components=keep1, thresh=keep2)

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    W = np.sqrt(1. / eigval0)  # diagonal of whitening matrix

    # c1 is projected into whitened PCA space of data channels
    c2 = (W * eigvec0.squeeze()).T.dot(c1).dot(eigvec0.squeeze()) * W

    # proj. matrix from whitened data space to a space maximizing bias
    eigvec2, eigval2 = pca(c2, max_components=keep1, thresh=keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = (W[np.newaxis, :] * eigvec0).dot(eigvec2)
    fromdss = linalg.pinv(todss)

    # Normalise DSS matrix
    N = np.sqrt(1. / np.diag(np.dot(np.dot(todss.T, c0), todss)))
    todss = todss * N

    pwr0 = np.sqrt(np.sum(np.dot(c0, todss) ** 2, axis=0))
    pwr1 = np.sqrt(np.sum(np.dot(c1, todss) ** 2, axis=0))

    # Return data
    # next line equiv. to: np.array([np.dot(todss, ep) for ep in data])
    # dss_data = np.einsum('ij,hjk->hik', todss, data)

    return todss, fromdss, pwr0, pwr1
