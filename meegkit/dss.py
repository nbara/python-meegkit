import numpy as np
from scipy import linalg
from .utils import demean, tscov, mean_over_trials, pcarot, theshapeof


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
    todss: array
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
    todss: array
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
    topcs1, evs1 = pcarot(c0, keep=keep1)
    if keep1:
        topcs1 = topcs1[:, np.arange(keep1)]
        evs1 = evs1[np.arange(keep1)]

    if keep2:
        idx = np.where(evs1 / np.max(evs1) > keep2)
        topcs1 = topcs1[:, idx]
        evs1 = evs1[idx]

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    N = np.diag(np.sqrt(1. / evs1))
    c2 = np.dot(np.dot(np.dot(np.dot(N.T, topcs1.squeeze().T), c1),
                topcs1.squeeze()), N)

    # derive the DSS matrix
    topcs2, evs2 = pcarot(c2, keep=keep1)

    # DSS matrix (raw data to normalized DSS)
    todss = np.dot(np.dot(topcs1.squeeze(), N), topcs2)
    fromdss = linalg.pinv(todss)

    # estimate power per DSS component
    # pwr = np.zeros((todss.shape[1], 1))
    # for k in range(todss.shape[1]):
    #     to_component = todss[:, k] * fromdss[k, :]
    #     cc = to_component.T * c0 * to_component
    #     cc = np.diag(cc)
    #     pwr[k] = np.sum(cc ** 2)
    # ratio = (np.diag(np.dot(np.dot(todss.T, c1), todss)) /
    #          np.diag(np.dot(np.dot(todss.T, c0), todss)))

    N2 = np.diag(np.dot(np.dot(todss.T, c0), todss))
    todss = np.dot(todss, np.diag(1. / np.sqrt(N2)))

    pwr0 = np.sqrt(np.sum(np.dot(c0, todss) ** 2))
    pwr1 = np.sqrt(np.sum(np.dot(c1, todss) ** 2))

    return todss, fromdss, pwr0, pwr1
