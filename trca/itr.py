import numpy as np

def itrFunc(n, p, t):
    """
    Calculate information transfer rate (ITR) for brain-computer interface 
    (BCI) [2]
    function [ itr ] = itr(n, p, t)
    
    Input:
    n   :  of targets
    p   : Target identification accuracy (0 <= p <= 1) 
    t   : Averaged time for a selection [s]
    
    Output:
    itr : Information transfer rate [bits/min] 
    
    Reference:
    [1] M. Cheng, X. Gao, S. Gao, and D. Xu,
        "Design and Implementation of a Brain-Computer Interface With High 
            Transfer Rates",
        IEEE Trans. Biomed. Eng. 49, 1181-1186, 2002.
    
    Code based on the Matlab implementation from https://github.com/mnakanishi/TRCA-SSVEP translated and adapted in Python by:

    Giuseppe Ferraro
    ISAE-SUPAERO
    github: gferraro2019 
    email: giuseppe.ferraro@isae.supaero.fr
    """
    itr=0

    if (p < 0 or 1 < p):
        print('stats:itr:BadInputValue.Accuracy need to be between 0 and 1.')  
    elif (p < 1/n):
        print("stats:itr:BadInputValue. The ITR might be incorrect because the accuracy < chance level.")
        itr = 0
    elif (p == 1):
        itr = np.log2(n)*60/t
    else:
        itr = (np.log2(n) + p*np.log2(p) + (1-p)*np.log2((1-p)/(n-1)))*60/t
    
    return itr