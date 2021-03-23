import math
import numpy as np
import scipy 

def round_half_up(num, decimals=0):
    """
    round_half_up round the last decimal of the number.

    The rules are:
    from 0 to 4 rounds down
    from 5 to 9 rounds up

    Parameters
    ----------
    num: number to round
    decimals : number of decimals

    Returns
    ---------
    num rounded
    """ 
    multiplier = 10 ** decimals
    return int(math.floor(num*multiplier + 0.5) / multiplier)

def normfit(data, confidence=0.95):
    """
    normfit compute the mean, std and interval of confidence for them.

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
    arr = 1.0 * np.array(data)
    num = len(arr)
    avg, std_err = np.mean(arr), scipy.stats.sem(arr)
    h_int = std_err * scipy.stats.t.ppf((1 + confidence) / 2., num - 1)
    var = np.var(data, ddof=1)
    var_ci_upper = var * (num - 1) / (scipy.stats.chi2.ppf((1-confidence) / 2, num - 1))
    var_ci_lower = var * (num - 1) / (scipy.stats.chi2.ppf(1-(1-confidence) / 2, num - 1))
    sigma = np.sqrt(var)
    sigma_ci_lower = np.sqrt(var_ci_lower)
    sigma_ci_upper = np.sqrt(var_ci_upper)

    return avg, sigma, [avg - h_int, avg + h_int], [sigma_ci_lower, sigma_ci_upper]
