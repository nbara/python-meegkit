"""TRCA utils."""
import numpy as np

from scipy.signal import filtfilt, cheb1ord, cheby1
from scipy import stats


def round_half_up(num, decimals=0):
    """Round half up round the last decimal of the number.

    The rules are:
    from 0 to 4 rounds down
    from 5 to 9 rounds up

    Parameters
    ----------
    num : float
        Number to round
    decimals : number of decimals

    Returns
    -------
    num rounded
    """
    multiplier = 10 ** decimals
    return int(np.floor(num * multiplier + 0.5) / multiplier)


def normfit(data, ci=0.95):
    """Compute the mean, std and confidence interval for them.

    Parameters
    ----------
    data : array, shape=()
        Input data.
    ci : float
        Confidence interval (default=0.95).

    Returns
    -------
    m : float
        Mean.
    sigma : float
        Standard deviation
    [m - h, m + h] : list
        Confidence interval of the mean.
    [sigmaCI_lower, sigmaCI_upper] : list
        Confidence interval of the std.
    """
    arr = 1.0 * np.array(data)
    num = len(arr)
    avg, std_err = np.mean(arr), stats.sem(arr)
    h_int = std_err * stats.t.ppf((1 + ci) / 2., num - 1)
    var = np.var(data, ddof=1)
    var_ci_upper = var * (num - 1) / stats.chi2.ppf((1 - ci) / 2, num - 1)
    var_ci_lower = var * (num - 1) / stats.chi2.ppf(1 - (1 - ci) / 2, num - 1)
    sigma = np.sqrt(var)
    sigma_ci_lower = np.sqrt(var_ci_lower)
    sigma_ci_upper = np.sqrt(var_ci_upper)

    return avg, sigma, [avg - h_int, avg +
                        h_int], [sigma_ci_lower, sigma_ci_upper]


def itr(n, p, t):
    """Compute information transfer rate (ITR).

    Definition in [1]_.

    Parameters
    ----------
    n : int
        Number of targets.
    p : float
        Target identification accuracy (0 <= p <= 1).
    t : float
        Average time for a selection (s).

    Returns
    -------
    itr : float
        Information transfer rate [bits/min]

    References
    ----------
    .. [1] M. Cheng, X. Gao, S. Gao, and D. Xu,
        "Design and Implementation of a Brain-Computer Interface With High
        Transfer Rates", IEEE Trans. Biomed. Eng. 49, 1181-1186, 2002.

    """
    itr = 0

    if (p < 0 or 1 < p):
        raise ValueError('Accuracy need to be between 0 and 1.')
    elif (p < 1 / n):
        itr = 0
        raise ValueError('ITR might be incorrect because accuracy < chance')
    elif (p == 1):
        itr = np.log2(n) * 60 / t
    else:
        itr = (np.log2(n) + p * np.log2(p) + (1 - p) *
               np.log2((1 - p) / (n - 1))) * 60 / t

    return itr


def bandpass(eeg, sfreq, Wp, Ws):
    """Filter bank design for decomposing EEG data into sub-band components.

    Parameters
    ----------
    eeg : np.array, shape=(n_samples, n_chans[, n_trials])
        Training data.
    sfreq : int
        Sampling frequency of the data.
    Wp : 2-tuple
        Passband for Chebyshev filter.
    Ws : 2-tuple
        Stopband for Chebyshev filter.

    Returns
    -------
    y: np.array, shape=(n_trials, n_chans, n_samples)
        Sub-band components decomposed by a filter bank.

    See Also
    --------
    scipy.signal.cheb1ord :
        Chebyshev type I filter order selection.

    """
    # Chebyshev type I filter order selection.
    N, Wn = cheb1ord(Wp, Ws, 3, 40, fs=sfreq)

    # Chebyshev type I filter design
    B, A = cheby1(N, 0.5, Wn, btype="bandpass", fs=sfreq)

    # the arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)'
    # correspond to Matlab filtfilt : https://dsp.stackexchange.com/a/47945
    y = filtfilt(B, A, eeg, axis=0, padtype='odd',
                 padlen=3 * (max(len(B), len(A)) - 1))
    return y


def schaefer_strimmer_cov(X):
    r"""Schaefer-Strimmer covariance estimator.

    Shrinkage estimator described in [1]_:

    .. math:: \hat{\Sigma} = (1 - \gamma)\Sigma_{scm} + \gamma T

    where :math:`T` is the diagonal target matrix:

    .. math:: T_{i,j} = \{ \Sigma_{scm}^{ii} \text{if} i = j,
         0 \text{otherwise} \}

    Note that the optimal :math:`\gamma` is estimated by the authors' method.

    Parameters
    ----------
    X: array, shape=(n_chans, n_samples)
        Signal matrix.

    Returns
    -------
    cov: array, shape=(n_chans, n_chans)
        Schaefer-Strimmer shrinkage covariance matrix.

    References
    ----------
    .. [1] Schafer, J., and K. Strimmer. 2005. A shrinkage approach to
       large-scale covariance estimation and implications for functional
       genomics. Statist. Appl. Genet. Mol. Biol. 4:32.
    """
    ns = X.shape[1]
    C_scm = np.cov(X, ddof=0)
    X_c = X - np.tile(X.mean(axis=1), [ns, 1]).T

    # Compute optimal gamma, the weigthing between SCM and srinkage estimator
    R = ns / (ns - 1.0) * np.corrcoef(X)
    var_R = (X_c ** 2).dot((X_c ** 2).T) - 2 * C_scm * X_c.dot(X_c.T)
    var_R += ns * C_scm ** 2

    var_R = ns / ((ns - 1) ** 3 * np.outer(X.var(1), X.var(1))) * var_R
    R -= np.diag(np.diag(R))
    var_R -= np.diag(np.diag(var_R))
    gamma = max(0, min(1, var_R.sum() / (R ** 2).sum()))

    cov = (1. - gamma) * (ns / (ns - 1.)) * C_scm
    cov += gamma * (ns / (ns - 1.)) * np.diag(np.diag(C_scm))

    return cov
