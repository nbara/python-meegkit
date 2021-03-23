import math
import numpy as np
import scipy 

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(math.floor(n*multiplier + 0.5) / multiplier)

def normfit(data, confidence=0.95):
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