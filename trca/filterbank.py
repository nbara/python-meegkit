import numpy as np
import scipy.signal as scp


def filterbank(eeg, fs, idx_fb):
    """
    Filter bank design for decomposing EEG data into sub-band components [1]
    
    Parameters
    ----------
    
    eeg: np.array, shape (trials, channels, samples)
        Training data 
        
    fs: int
        Sampling frequency of the data.
    
    idx_fb: int
        Index of filters in filter bank analysis
            
    Returns
    -------
    
    y: np.array, shape (trials, channels, samples)
        Sub-band components decomposed by a filter bank

    Reference:
      [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using 
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.
    
    Code based on the Matlab implementation from https://github.com/mnakanishi/TRCA-SSVEP translated and adapted in Python by:

    Giuseppe Ferraro
    ISAE-SUPAERO
    github: gferraro2019 
    email: giuseppe.ferraro@isae.supaero.fr
    """
    if(eeg.ndim == 3):
        num_chans = eeg.shape[1]
        num_trials = eeg.shape[0]


    elif(eeg.ndim == 2): #  Testdata come with only one trial at the time
        num_chans = eeg.shape[0]
        num_trials = 1
    

    fs=fs/2

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb]/fs, 90/fs]
    Ws = [stopband[idx_fb]/fs, 100/fs]
    N, Wn = scp.cheb1ord(Wp, Ws, 3, 40)  # Chebyshev type I filter order selection.
    
    B, A = scp.cheby1(N, 0.5, Wn,btype="bandpass")  #  Chebyshev type I filter desing

    y = np.zeros(eeg.shape) 
    if num_trials == 1:  # For testdata
        for ch_i in range(num_chans):
            try:
                y[ch_i, :] = scp.filtfilt(B, A, eeg[ch_i, :], axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1))
            except:
                print(num_chans)
    else:
        for trial_i in range(num_trials):  # Filter each trial sequentially 
            for ch_i in range(num_chans):  # Filter each channel sequentially 
                # the arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)' correspond to Matlab filtfilt (https://dsp.stackexchange.com/a/47945)
                y[trial_i, ch_i, :] = scp.filtfilt(B, A, eeg[trial_i, ch_i, :], axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1))
    return y
