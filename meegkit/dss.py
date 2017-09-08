import numpy as np
# import scipy.linalg
from .utils import demean, tscov, mean_over_trials, pcarot


def dss1(data, weights=None, keep1=None, keep2=None):
    """DSS to maximise repeatability across trials."""
    if not any(weights):
        weights = np.array([])
    if not keep1:
        keep1 = np.array([])
    if not keep2:
        keep2 = 10.0 ** -12

    m, n, o = data.shape()
    data, data_mean = demean(data, weights)  # remove weighted mean

    # weighted mean over trials (--> bias function for DSS)
    xx, ww = mean_over_trials(data, weights)
    print("xx.shape", xx.shape)
    ww = ww.min(1)

    # covariance of raw and biased data
    c0, nc0 = tscov(data, None, weights)
    c1, nc1 = tscov(xx, None, ww)
    c1 = c1 / o

    todss, fromdss, ratio, pwr = dss0(c0, c1, keep1, keep2)

    return todss, fromdss, ratio, pwr


def dss0(c1, c2, keep1, keep2):
    """DSS base function."""
    # SANITY CHECKS GO HERE

    # derive PCA and whitening matrix from unbiased covariance
    topcs1, evs1 = pcarot(c1)
    if keep1:
        topcs1 = topcs1[:, np.arange(keep1)]
        evs1 = evs1[np.arange(keep1)]

    if keep2:
        idx = np.where(evs1 / np.max(evs1) > keep2)
        topcs1 = topcs1[:, idx]
        evs1 = evs1[idx]

    # apply whitening and PCA matrices to the biased covariance
    # (== covariance of bias whitened data)
    N = np.diag(np.sqrt(1 / evs1))
    c3 = np.dot(np.dot(np.dot(np.dot(N.T, topcs1.squeeze().T), c2),
                topcs1.squeeze()), N)

    # derive the dss matrix
    topcs2, evs2 = pcarot(c3)
    todss = topcs1.squeeze() * N * topcs2
    fromdss = np.linalg.pinv(todss)

    # estimate power per DSS component
    pwr = np.zeros((todss.shape[1], 1))

    for k in range(todss.shape[1]):
        to_component = todss[:, k] * fromdss[k, :]
        cc = to_component.T * c1 * to_component
        cc = np.diag(cc)
        pwr[k] = sum(cc**2)

    ratio = np.diag(todss.T * c2 * todss) / np.diag(todss.T * c1 * todss)

    return todss, fromdss, ratio, pwr
