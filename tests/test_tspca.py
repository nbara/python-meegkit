import numpy as np

from context import meegkit  # noqa
from meegkit import dss, sns, tspca
from meegkit.utils import demean, fold, unfold


def test_tspca_sns_dss():
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
    data = np.random.random((800, 102, 200))
    ref = np.random.random((800, 3, 200))

    # remove means
    noisy_data, _ = demean(data)
    noisy_ref, _ = demean(ref)

    # Apply TSPCA
    # -------------------------------------------------------------------------
    # shifts = np.r_[-50:51]
    # print('TSPCA...')
    # y_tspca, idx = tspca.tsr(noisy_data, noisy_ref, shifts)[0:2]
    # print('\b OK!')
    y_tspca = noisy_data

    # Apply SNS
    # -------------------------------------------------------------------------
    nneighbors = 10
    print('SNS...')
    y_tspca_sns, r = sns.sns(y_tspca, nneighbors)
    print('\b OK!')

    # apply DSS
    # -------------------------------------------------------------------------
    print('DSS...')
    # Keep all PC components
    y_tspca_sns, _ = demean(y_tspca_sns)
    print(y_tspca_sns.shape)
    todss, fromdss, _, _ = dss.dss1(y_tspca_sns)
    print('\b OK!')

    # c3 = DSS components
    y_tspca_sns_dss = fold(
        np.dot(unfold(y_tspca_sns), todss), y_tspca_sns.shape[0])

    return y_tspca, y_tspca_sns, y_tspca_sns_dss


if __name__ == '__main__':
    import nose
    nose.run(defaultTest="")
