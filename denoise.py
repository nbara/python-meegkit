from numpy import *
import scipy.linalg

#data = random.random((800,157,241))
#ref  = random.random((800,3,241))

def multishift(data, shifts, amplitudes = array([])):
    """apply multiple shifts to an array"""
    #print "multishift"
    if min(shifts) > 0: raise Exception('shifts should be non-negative')
        
    shifts = shifts.T
    shifts_length = shifts.size
    
    # array of shift indices
    
    if data.ndim == 3:
        time, channels, trials = data.shape # m, n, o
    elif data.ndim == 2:
        time, channels = data.shape
        trials = 1
    else:
        time = data.shape
        channels, trials = 1, 1
    
    N = time - max(shifts)
    shiftarray = ((ones((N, shifts_length), int) * shifts).T + r_[ 0:N ]).T
    
    z = zeros((N, channels * shifts_length, trials))
    
    if amplitudes:
        for trial in arange(trials):
            for channel in arange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = (y[shiftarray].T * amplitudes).T
    else:
        for trial in arange(trials):
            for channel in arange(channels):
                y = data[:, channel]
                z[:, (channel * shifts_length):(channel * shifts_length + shifts_length), trial] = y[shiftarray]
        
                
    return z


def pcarot(cov, keep = None):
    """
    PCA rotation from covariance.
    
    INPUT
    cov:  covariance matrix
    keep: number of components to keep [default: all]
    
    OUTPUT
    topcs:       PCA rotation matrix
    eigenvalues: PCA eigenvalues
    """
    
    if not keep: 
        keep = cov.shape[0] # keep all components
        
    eigenvalues, eigenvector = linalg.eig(cov)
    
    idx = argsort(eigenvalues.real)[::-1] # reverse sort ev order
    eigenvalues = sort(eigenvalues.real)[::-1]
    
    topcs = eigenvector.real[:, idx]
    
    topcs = topcs[:, arange(keep)]
    eigenvalues = eigenvalues[arange(keep)]
    
    return topcs, eigenvalues


def tscov(data, shifts = None, weights = None):
    """
    Time shift covariance.
    
    This function calculates, for each pair [DATA[i], DATA[j]] of columns of 
    DATA, the cross-covariance matrix between the time-shifted versions of 
    DATA[i]. Shifts are taken from array SHIFTS. Weights are taken from WEIGHTS.
    
    DATA can be 1D, 2D or 3D.  WEIGHTS is 1D (if DATA is 1D or 2D) or 
    2D (if DATA is 3D).
    
    Output is a 2D matrix with dimensions (ncols(X)*nshifts)^2.
    This matrix is made up of a DATA.shape[1]^2 matrix of submatrices
    of dimensions nshifts^2.
    
    The weights are not shifted.
    
    INPUT
    data: data
    shifts: array of time shifts (must be non-negative)
    weights: weights
    
    OUTPUT
    covariance_matrix: covariance matrix
    total_weight: total weight (covariance_matrix/total_weight is normalized 
    covariance)
    """
    
    if shifts == None:  
        shifts  = array([0])
    if not any(weights): 
        weights = array([])
    
    if shifts.min() < 0: raise ValueError, "Shifts should be non-negative."
    
    shifts = shifts.T
    nshifts = shifts.size
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
    else:
        samples, channels = data.shape
        trials = 1
    
    covariance_matrix = zeros((channels * nshifts, channels * nshifts))
    
    if any(weights):
        # weights
        if weights.shape[1] > 1: 
            raise ValueError, "Weights array should have a single column"
        
        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = multishift(data[:, :, trial], shifts)
            else:
                shifted_trial = multishift(data[:, trial], shifts)
            
            trial_weight = weights[arange(shifted_trial.shape[0]), :, trial]
            shifted_trial = (squeeze(shifted_trial).T * squeeze(trial_weight)).T
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
        
        total_weight = sum(weights[:])
    else:
        # no weights
        for trial in xrange(trials):
            if data.ndim == 3:
                shifted_trial = squeeze(multishift(data[:, :, trial], shifts))
            else:
                shifted_trial = multishift(data[:, trial], shifts)
            
            covariance_matrix += dot(shifted_trial.T, shifted_trial)
        
        total_weight = shifted_trial.shape[0] * trials
    
    return covariance_matrix, total_weight


def fold(data, epochsize):
    '''fold'''
    return transpose(reshape(data, (epochsize, data.shape[0]/epochsize, data.shape[1])), (0, 2, 1))


def unfold(data):
    '''unfold'''
    try:
        samples, channels, trials = data.shape
    except:
        samples, channels = data.shape
        trials = 1
    
    if trials > 1:
        return reshape(transpose(data, (0, 2, 1)), (samples * trials, channels))
    else:
        return data


def demean(data, weights = None):
    """Remove weighted mean over columns."""
    
    if data.ndim == 3:
        samples, channels, trials = data.shape
    else:
        samples, channels = data.shape
    
    data = unfold(data)
    
    if any(weights):
        weights = unfold(weights)
        
        if weights.shape[0] != data.shape[0]:
            raise ValueError, "Data and weights arrays should have same number of rows and pages."
        
        if weights.shape[1] == 1 or weights.shape[1] == channels:
            the_mean = sum(data * weights) / sum(weights)
        else:
            raise ValueError, "Weight array should have either the same number of columns as data array, or 1 column."
        
        demeaned_data = data - the_mean
    else:
        the_mean = mean(data, 0)
        demeaned_data = data - the_mean
    
    demeaned_data = fold(demeaned_data, samples)
    
    return demeaned_data, the_mean


def normcol(data, weights = None):
    """
    Normalize each column so its weighted msq is 1.
    
    If DATA is 3D, pages are concatenated vertically before calculating the 
    norm.
    
    Weight should be either a column vector, or a matrix (2D or 3D) of same size
    as data.
    
    INPUT
    data: data to normalize
    weights: weight
    
    OUTPUT
    normalized_data: normalized data
    
    """
        
    if data.ndim == 3:
        samples, channels, trials = data.shape
        data = unfold(data)
        if not weights.any():
            # no weights
            normalized_data = fold(normcol(data), samples)
        else:
            if weights.shape[0] != samples: 
                raise ValueError, "Weight array should have same number of columns as data array."
            
            if weights.ndim == 2 and weights.shape[1] == 1: 
                weights = tile(weights, (1, samples, trials))
            
            if weights.shape != weights.shape: 
                raise ValueError, "Weight array should have be same shape as data array"
            
            weights = unfold(weights)
            
            normalized_data = fold(normcol(data, weights), samples)
    else:
        samples, channels = data.shape
        if not weights.any():
            normalized_data = data * ((sum(data ** 2) / samples) ** -0.5)
        else:
            if weights.shape[0] != data.shape[0]: 
                raise ValueError, "Weight array should have same number of columns as data array."
            
            if weights.ndim == 2 and weights.shape[1] == 1: 
                weights = tile(weights, (1, channels))
            
            if weights.shape != data.shape: 
                raise ValueError, "Weight array should have be same shape as data array"
            
            if weights.shape[1] == 1: 
                weights = tile(weights, (1, channels))
            
            normalized_data = data * (sum((data ** 2) * weights) / sum(weights)) ** -0.5
    
    return normalized_data


def regcov(cxy, cyy, keep=array([]), threshold=array([])):
    """regression matrix from cross covariance"""
    
    # PCA of regressor
    [topcs, eigenvalues] = pcarot(cyy)
    
    # discard negligible regressor PCs
    if keep:
        keep = max(keep, topcs.shape[1])
        topcs = topcs[:, 0:keep]
        eigenvalues = eigenvalues[0:keep]
    
    if threshold:
        idx = where(eigenvalues/max(eigenvalues) > threshold)
        topcs = topcs[:, idx]
        eigenvalues = eigenvalues[idx]
    
    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = dot(topcs.T, cxy)
    
    # projection matrix from regressor PCs
    r = (r.T * 1/eigenvalues).T
    
    #projection matrix from regressors
    r = dot(squeeze(topcs), squeeze(r))
    
    return r


def tsxcov(x, y, shifts = None, w = array([])):
    """
    Calculate cross-covariance of X and time-shifted Y.
    
    This function calculates, for each pair of columns (Xi,Yj) of X and Y, the
    scalar products between Xi and time-shifted versions of Yj. 
    Shifts are taken from array SHIFTS. 
    
    The weights are applied to X.
    
    X can be 1D, 2D or 3D.  W is 1D (if X is 1D or 2D) or 2D (if X is 3D).
    
    Output is a 2D matrix with dimensions ncols(X)*(ncols(Y)*nshifts).
    
    INPUT
    x, y: data to cross correlate
    shifts: array of time shifts (must be non-negative)
    w: weights
    
    OUTPUT
    c: cross-covariance matrix
    tw: total weight
    """
    
    if shifts == None: 
        shifts = array([0])
        
    nshifts = shifts.size
    
    mx, nx, ox = x.shape
    my, ny, oy = y.shape
    c = zeros((nx, ny*nshifts))
    
    if any(w): 
        x = fold(unfold(x) * unfold(w), mx)
        
    # cross covariance
    for trial in xrange(ox):
        yy = squeeze(multishift(y[:, :, trial], shifts))
        xx = squeeze(x[0:yy.shape[0], :, trial])
        
        c += dot(xx.T, yy)
        
    if not any(w):
        tw = ox * ny * yy.shape[0]
    else:
        w = w[0:yy.shape[0], :, :]
        tw = sum(w[:])
        
    return c, tw


def tsregress(x, y, shifts = array([0]), keep = array([]), threshold = array([]), toobig1 = array([]), toobig2 = array([])):
    """docstring for tsregress"""
    
    # shifts must be non-negative
    mn = shifts.min()
    if mn < 0:
        shifts = shifts - mn
        x = x[-mn+1:, :, :]
        y = y[-mn+1:, :, :]
    
    nshifts = shifts.size
    
    # flag outliers in x and y
    if toobig1 or toobig2:
        xw = find_outliers(x, toobig1, toobig2)
        yw = find_outliers(y, toobig1, toobig2)
    else:
        xw = []
        yw = []
    
    if x.ndim == 3:
        [Mx, Nx, Ox] = x.shape
        [My, Ny, Oy] = y.shape
        x = unfold(x)
        y = unfold(y)
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)
        x = fold(x, Mx)
        y = fold(y, My)
    else:
        [x, xmn] = demean(x, xw)
        [y, ymn] = demean(y, yw)
    
    # covariance of y
    [cyy, totalweight] = tscov(y, shifts.T, yw)
    cyy = cyy / totalweight
    
    # cross-covariance of x and y
    [cxy, totalweight] = tscov2(x, y, shifts.T, xw, yw)
    cxy = cxy / totalweight
    
    # regression matrix
    r = regcov(cxy, cyy, keep, threshold)
    
    # regression
    if x.ndim == 3:
        x = unfold(x)
        y = unfold(y)
        
        [m, n, o] = x.shape
        mm = m - max(shifts)
        z = zeros(x.shape)
        
        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+mm]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]
        
        z = fold(z, Mx)
        z = z[0:-max(shifts), :, :]
    else:
        m, n = x.shape
        z = zeros((m-max(shifts), n))
        for k in xrange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+z.shape[0]]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z = z + y[idx1, :] * r[idx2, :]
    
    offset = max(0, -mn)
    idx = r_[offset+1:offset+z.shape[0]]
    
    return [z, idx]


def find_outliers(x, toobig1, toobig2 = []):
    """docstring for find_outliers"""
    
    [m, n, o] = x.shape
    x = unfold(x)
    
    # remove mean
    x = demean(x)[0]
    
    # apply absolute threshold
    w = ones(x.shape)
    if toobig1:
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x, w)[0]
    else:
        w = ones(x.shape)
    
    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, w)
        X = tile(X, (x.shape[0], 1))
        idx = where(x**2 > (X * toobig2))
        w[idx] = 0
    
    w = fold(w, m)
    
    return w


def wmean(x, w=[], dim=0):
    """docstring for wmean"""
    
    if not w:
        y = mean(x, dim)
    else:
        if x.shape[0] != w.shape[0]:
            raise Exception("data and weight must have same nrows")
        if w.shape[1] == 1:
            w = tile(w, (1, x.shape(1)))
        if w.shape[1] != x.shape[1]:
            raise Exception("weight must have same ncols as data, or 1")
        y = sum(x * w, dim) / sum(w, dim)
    
    return y



def mean_over_trials(x, w):
    """docstring for mean_over_trials"""
    
    m, n, o = x.shape
    
    if not any(w):
        y = mean(x, 2)
        tw = ones((m, n, 1)) * o
    else:
        mw, nw, ow = w.shape
        if mw != m: raise "!"
        if ow != o: raise "!"
        
        x = unfold(x)
        w = unfold(w)
        
        if nw == n:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) / sum(w, 3)
        elif nw == 1:
            x = x * w
            x = fold(x, m)
            w = fold(w, m)
            y = sum(x, 3) * 1/sum(w, 3)
        
        tw = sum(w, 3)
    
    return y, tw
    
            
            
            




