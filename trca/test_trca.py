import numpy as np
import scipy

from filterbank import filterbank


def test_trca(eeg, model, is_ensemble):
    """Test phase of the task-related component analysis (TRCA)-based
    steady-state visual evoked potentials (SSVEPs) detection [1].

    Parameters
    ----------
    
    eeg: np.array, shape (trials, channels, samples)
        Test data 
    
    model: dict
        Fitted model to be used in testing phase 

    is_ensemble: boolean
        Perform the ensemble TRCA analysis or not
            
    Returns
    -------
    
    results: np.array, shape (trials)
        The target estimated by the method

    Reference:
      [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using 
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.
    
    Code based on the Matlab implementation from https://github.com/mnakanishi/TRCA-SSVEP translated and adapted in Python by:

    Giuseppe Ferraro
    ISAE-SUPAERO
    github: gferraro2019 
    email: giuseppe.ferraro@isae.supaero.fr"""
    
    
    fb_coefs = [(x+1)**(-1.25)+0.25 for x in range(model["num_fbs"])] #  Alpha coefficients for the fusion of filterbank analysis
    testdata_len = len(eeg)
    
    r= np.zeros((model["num_fbs"],len(model["num_targs"])))
    results = np.zeros((testdata_len),'int')  # To store predictions

    for trial in range(testdata_len):
        test_tmp = eeg[trial, :, :] #  Pick a trial to be analysed 
        for fb_i in range(model["num_fbs"]):
            testdata = filterbank(test_tmp, model["fs"], fb_i)  # Filterbank on testdata
            for class_i in model["num_targs"]:
                traindata =  np.squeeze(model["trains"][class_i, fb_i, :, :]) #  Retrive reference signal for clss_i (shape: (# of channel, # of sample))
                if is_ensemble:
                    w = np.squeeze(model["W"][fb_i, :, :]).T  # Shape of (# of channel, # of class)
                else:
                    w = np.squeeze(model["W"][fb_i, class_i, :]) # Shape of (# of channel)
                r_tmp = np.corrcoef(np.dot(testdata.T,w).flatten(), np.dot(traindata.T,w).flatten())  # Compute 2D correlation of spatially filtered testdata with ref
                r[fb_i,class_i] = r_tmp[0,1]
            
        rho = np.dot(fb_coefs,r)  # Fusion for the filterbank analysis

        tau = np.argmax(rho) #  Retrieving the index of the max
        results[trial] = int(tau)

    
    return results 
