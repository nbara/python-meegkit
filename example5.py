from denoise import *
from tspca import *
from sns import *
from dss import *
from scipy.io import mio

#x = mio.loadmat('data2.mat')
#data = x['data']
#ref  = x['ref']

def tspca_sns_dss(data, ref):
    """
    Requires data stored in a time X channels X trials matrix.
    
    Remove environmental noise with TSPCA (shifts=-50:50).
    Remove sensor noise with SNS.
    Remove non-repeatable components with DSS.
    """
    
    data = random.random((800,157,200))
    ref = random.random((800,3,200))
    
    # remove means
    noisy_data = demean(data)[0]
    noisy_ref  = demean(ref)[0]
    
    # apply TSPCA
    shifts = r_[-50:51]
    print 'TSPCA ...'
    data_tspca, idx = tsr(noisy_data, noisy_ref, shifts)[0:2]
    
    ## apply SNS
    #nneighbors = 10
    #print 'SNS ...'
    #data_tspca_sns = sns(data_tspca, nneighbors)
    
    # apply DSS
    
    #disp('DSS ...');
    ## Keep all PC components
    #data_tspca_sns = demean(data_tspca_sns)[0]
    #todss, fromdss, ratio, pwr = dss1(data_tspca_sns)
    ## c3 = DSS components
    #data_tspca_sns_dss = fold(unfold(data_tspca_sns) * todss, data_tspca_sns.shape[0]); 
    
    return data_tspca

#tspca_sns_dss()