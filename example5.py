def tspca_sns_dss(data, ref, sr):
    """
    Requires data stored in a time X channels X trials matrix.

    Remove environmental noise with TSPCA (shifts=-50:50).
    Remove sensor noise with SNS.
    Remove non-repeatable components with DSS.
    """

    # remove means
    noisy_data = demean(data)[0]
    noisy_ref  = demean(ref)[0]
    
    # apply TSPCA
    shifts = r_[-50:50]
    print 'TSPCA ...'
    data_tspca, idx = tsr(noisy_data, noisy_ref, shifts)
    
    # apply SNS
    nneighbors = 10
    print 'SNS ...'
    data_tspca_sns = sns(data_tspca, nneighbors)
    
    # apply DSS
    
    disp('DSS ...');
    # Keep all PC components
    todss, fromdss, ratio, pwr = dss1(demean(data_tspca_sns[:,:,:]))
    # c3 = DSS components
    data_tspca_sns_dss = fold(unfold(demean(data_tspca_sns)) * todss, data_tspca_sns.shape[0]); 
    