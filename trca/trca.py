import numpy as np
import scipy.linalg as linalg
from joblib import Parallel, delayed

def trca(eeg):
    """Task-related component analysis (TRCA). This script was written based on
    the reference paper [1].
    
    Parameters
    ----------
    
    eeg: np.array, shape (trials, channels, samples)
        Training data 
            
    Returns
    -------
    
    W: np.array, shape (channels)
        Weight coefficients for electrodes which can be used as 
        a spatial filter.

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
    
    num_chans = eeg.shape[1]
    num_smpls = eeg.shape[2]

    if(eeg.ndim == 3):
        num_trials = eeg.shape[0] 

    elif(eeg.ndim == 2):  # For testdata
        num_trials = 1

    S = np.zeros((num_chans,num_chans))
    for trial_i in range(num_trials-1):
        x1 = np.squeeze(eeg[trial_i,:,:])
        if x1.ndim>1:
            x1 = x1 - (np.mean(x1,1)*np.ones((x1.shape[0],x1.shape[1])).T).T  # Mean centering for the selected trial
        else:
            x1 = x1 - np.mean(x1)

        for trial_j in range(trial_i+1,num_trials):  # Select a second trial that is different
            x2 = np.squeeze(eeg[trial_j,:,:])
            if x2.ndim>1:
                x2 = x2 - (np.mean(x2,1)*np.ones((x2.shape[0],x2.shape[1])).T).T  # Mean centering for the selected trial
            else:
                x2 = x2 - np.mean(x2)
            
            S = S + np.dot(x1,x2.T) + np.dot(x2,x1.T)  # Compute empirical covariance betwwen the two selected trials and sum it

    UX = np.zeros(( num_chans, num_smpls*num_trials))  # Reshape to have all the data as a sequence
    for trial in range(num_trials):
        UX[:,trial*num_smpls:(trial+1)*num_smpls]=eeg[trial,:,:]

    UX = UX - (np.mean(UX,1)*np.ones((UX.shape[0],UX.shape[1])).T).T  # Mean centering
    Q =  np.dot(UX,UX.T)  #  Compute empirical variance of all data (to be bounded)

    lambdas,W = linalg.eig(S, Q,left=True, right=False)  # Compute eigenvalues and vectors
    W_best = W[:,np.argmax(lambdas)]  # Select the eigenvector coresponding to the biggest eigenvalue

    return W_best
