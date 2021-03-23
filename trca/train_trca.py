import numpy as np

from  filterbank import filterbank 
from  trca import trca


def train_trca(eeg, y_train, fs, num_fbs): 
    """Training stage of the task-related component analysis (TRCA)-based
    steady-state visual evoked potentials (SSVEPs) detection [1].
    
    Parameters
    ----------
    
    eeg: np.array, shape (trials, channels, samples)
        Training data 
          
    y_train: list or np.array, shape (trials)
        True label corresponding to each trial of the data
        array.
        
    fs: int
        Sampling frequency of the data.
    
    num_fb: int
        Number of sub-bands considered for the filterbank analysis
            
    Returns
    -------
    
    model: dict
        Fitted model containing:
        - traindata   : Reference (training) data decomposed into sub-band components 
                        by the filter bank analysis
                        (# of trials, # of sub-bands, # of channels, 
                         Data length [sample])
        - y_train     : Labels associated with the train data
                        (# of trials)
        - W           : Weight coefficients for electrodes which can be 
                        used as a spatial filter.
        - num_fbs     : # of sub-bands
        - fs          : Sampling rate
        - num_targs   : # of targets

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

    num_class = np.unique(y_train)
    
    trains = np.zeros((len(num_class), num_fbs, num_chans, num_smpls))  

    W = np.zeros((num_fbs, len(num_class), num_chans))

    for class_i in num_class:
        eeg_tmp = eeg[y_train==class_i] # Select data with a specific label
        for fb_i in range(num_fbs):
            eeg_tmp = filterbank(eeg_tmp, fs, fb_i)  # Filter the signal with fb_i
            if(eeg_tmp.ndim == 3):
                trains[class_i,fb_i,:,:] = np.mean(eeg_tmp, 0)  # Compute mean of the signal accross the trials
            else:
                trains[class_i,fb_i,:,:] = eeg_tmp
            w_best= trca(eeg_tmp)  # Find the spatial filter for the corresponding filtered signal and label
            W[fb_i, class_i, :] = w_best  # Store the spatial filter

        
    model = {'trains': trains, 
               'W': W,
               'num_fbs': num_fbs,
               'fs': fs,
             'num_targs': num_class}
    return model
