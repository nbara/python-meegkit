import math
import numpy as np
import scipy 

def round_half_up(n, decimals=0):
    """
    This function rounds the last decimal of the number as below:
    from 0 to 4 rounds down
    from 5 to 9 rounds up

    Parameters
    ----------
    n: number to round
    decimals : number of decimals

    Returns
    ---------
    n rounded
    """ 
    multiplier = 10 ** decimals
    return int(math.floor(n*multiplier + 0.5) / multiplier)

def normfit(data, confidence=0.95):
    """
    Parameters
    ----------
    data: array of numbers
    confidence : interval of confidence. Default = 0.95.

    Returns
    -------
    m : mean
    sigma : std deviation
    [m - h, m + h] : confidence interval of the mean
    [sigmaCI_lower, sigmaCI_upper] : confidence interval of the std
    """

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    var = np.var(data, ddof=1)
    varCI_upper = var * (n - 1) / (scipy.stats.chi2.ppf((1-confidence) / 2, n - 1))
    varCI_lower = var * (n - 1) / (scipy.stats.chi2.ppf(1-(1-confidence) / 2, n - 1))
    sigma = np.sqrt(var)
    sigmaCI_lower = np.sqrt(varCI_lower)
    sigmaCI_upper = np.sqrt(varCI_upper)

    return m, sigma, [m - h, m + h], [sigmaCI_lower, sigmaCI_upper]