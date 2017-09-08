import numpy as np
from meegkit import sns, dss, tspca
from meegkit.utils import demean, fold, unfold


def tspca_sns_dss(data=None, ref=None):
    """Test TSPCA, SNS, DSS.

    Requires data stored in a time X channels X trials matrix.

    Remove environmental noise with TSPCA (shifts=-50:50).
    Remove sensor noise with SNS.
    Remove non-repeatable components with DSS.
    """
    # x = mio.loadmat('data2.mat')
    # data = x['data']
    # ref  = x['ref']

    # Random data (time*chans*trials)
    data = np.random.random((800, 157, 200))
    ref = np.random.random((800, 3, 200))

    # remove means
    noisy_data = np.demean(data)[0]
    noisy_ref = np.demean(ref)[0]

    # apply TSPCA
    shifts = np.r_[-50:51]
    print('TSPCA...')
    y_tspca, idx = tspca.tsr(noisy_data, noisy_ref, shifts)[0:2]

    # apply SNS
    nneighbors = 10
    print('SNS...')
    y_tspca_sns = sns.sns(y_tspca, nneighbors)

    # apply DSS
    print('DSS...')
    # Keep all PC components
    y_tspca_sns = demean(y_tspca_sns)[0]
    todss, fromdss, ratio, pwr = dss.dss1(y_tspca_sns)
    # c3 = DSS components
    y_tspca_sns_dss = fold(unfold(y_tspca_sns) * todss,
                           y_tspca_sns.shape[0])

    return y_tspca, y_tspca_sns, y_tspca_sns_dss


dat = tspca_sns_dss()
