import numpy as np
from scipy.io import mio

from meegkit import dss, sns

from ..utils import demean, fold, tsr, unfold, wpwr


def clean(data, ref):
    """Clean data using TSPCA, SNS, and DSS.

    Remove environmental noise with TSPCA (shifts=-50:50).
    Remove sensor noise with SNS.
    Remove non-repeatable components with DSS.

    Requires data stored in a time X channels X trials matrix.

    """
    # remove means
    noisy_data = demean(data)[0]
    noisy_ref = demean(ref)[0]

    # apply TSPCA
    shifts = np.arange(-50, 51)
    print('TSPCA ...')
    data_tspca, idx = tsr(noisy_data, noisy_ref, shifts)[0:2]

    data = data[idx, :, :]
    data_mean = np.mean(data, 2)
    data_tspca_mean = np.mean(data_tspca, 2)

    # Stats
    p1 = wpwr(data)[0]
    pp1 = wpwr(data_mean)[0]
    p2 = wpwr(data_tspca)[0]
    pp2 = wpwr(data_tspca_mean)[0]

    print("TSPCA done. {} of raw power remains".format(100 * p2 / p1))
    # print "trial-averaged: ", 100*pp2/pp1, " of raw power remains"

    # apply SNS
    nneighbors = 10
    print('SNS ...')
    data_tspca_sns = sns(data_tspca, nneighbors)

    # apply DSS
    print('DSS ...')
    # Keep all PC components
    data_tspca_sns = demean(data_tspca_sns)[0]
    todss, fromdss, ratio, pwr = dss.dss1(data_tspca_sns)
    # c3 = DSS components
    data_tspca_sns_dss = fold(
        np.dot(unfold(data_tspca_sns), todss), data_tspca_sns.shape[0])

    return data_tspca_sns


x = mio.loadmat('data2.mat')
data = x['data']
ref = x['ref']

cleandata = clean(data, ref)
