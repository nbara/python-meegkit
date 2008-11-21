from numpy import *
from numpy.random import permutation
import scipy as Sci
import scipy.linalg

def tspca(x, shifts=[0], keep=[], threshold=[], w=[]):
    """TSPCA"""
    
    [m, n, o] = x.shape
    
    # offset of z relative to x
    offset = max(0, -min(shifts, 0))
    shifts += offset
    idx = offset + (r_[1:m] - max([shifts]))
    
    # remove mean
    x = fold(demean(unfold(x), w), m)
    
    # covariance
    if w:
        c = tscov(x, shifts)
    else:
        if sum(w) == 0:
            raise Exception('weights are all zero')
        c = tscov(x, shifts, w);
        
    # PCA matrix
    [topcs, evs] = pcarot(c)
    
    # truncate
    if not keep:
        topcs = topcs[:, r_[1:keep+1]]
        evs = evs[r_[1:keep+1]]
        
    if not threshold:
        ii = where(evs/evs[0] > threshold)
        topcs = topcs[:, ii]
        evs = evs[ii]
    
    # apply PCA matrix to time-shifted data    
    z = zeros(idx.size, topcs.shape[2], o)
    
    for k in r_[1:o+1]:
        z[:, :, k] = multishift(x[:,:,k], shifts) * topcs
        
    return [z, idx]
        

def multishift(x, shifts, amplitudes=[]):
    """multishift"""
    if min(shifts) > 0: raise Exception('shifts should be non-negative')
        
    shifts = shifts.T
    nshifts = shifts.size
    
    # array of shift indices
    N = x.shape[0] - max(shifts)
    shiftarray = (ones((N, nshifts)) * shifts) + r_[1:N+1].T
    [m, n, o] = x.shape
    z = zeros((N, n*nshifts, o))
    
    if amplitudes:
        for k in arange(o):
            for j in arange(n):
                y = x[:, j+1] # this might be x[:, j+2]
                z[:, j*nshifts+1:j*nshifts+nshifts,k] = (y[shiftarray] * amplitudes)
    
    return z
                    

def pcarot(cov, keep=[]):
    """docstring for pcarot"""
    
    if not keep:
        keep = cov.shape[0]
    
    [V, S] = linalg.eig(cov)[0]
    V = V.real
    S = S.real
    
    [eigenvalues, idx] = sort(diag(S).T)
    eigenvalues = fliplr(idx)
    topcs = V[:,idx]
    
    eigenvalues = eigenvalues[r_[0:keep]]
    topcs = topcs[:,r_[0:keep]]
    

def tscov(x, shifts=0, w=[]):
    """docstring for tscov"""
    if min(shifts) < 0:
        raise Exception('shifts should be non-negative')
        
    nshifts = shifts.size
    
    [m,n,o] = x.shape
    c = zeros(n*shifts)
    
    if w:
        if w.shape[1] > 1: raise Exception('w should have a single column')
            
        for k in arange(o):
            xx = multishift(x[:,:,k], shifts)
            ww = w[arange(xx.shape[0]), :, k]
            xx = xx * ww
            c = c + xx.T * xx
        
        tw = sum(w[:])
    else:
        for k in range(o):
            xx = multishift(x[:,:,k], shifts)
            c = c + xx.T * xx
            
        tw = xx.shape[0] * o

    return c, tw
    

def fold(x, epochsize):
    """docstring for fold"""
    return transpose(reshape(x, (epochsize, x.shape[0]/epochsize, x.shape[1])), (0, 2, 1))


def unfold(x):
    """docstring for unfold"""
    [m, n, p] = x.shape
    
    if p > 1:
        return reshape(transpose(x, (0, 2, 1)), (m*p, n))
    else:
        return x

def demean(x, w=[]):
    """docstring for demean"""
    [m,n,o] = x.shape
    x = unfold(x)
    
    if not w:
        mn = mean(x,0)
        y = x - mn
    else:
        w = unfold(w)
        
        if w.shape[0] != x.shape[0]:
            raise Exception('X and W should have same nrows & npages')
        
        if w.shape[1] == 1:
            mn = sum(x * w) / sum(w,0)
        elif w.shape[1] == n:
            mn = sum(x * w) / sum(w,0)
        else:
            raise Exception('W should have same number of cols ans X, or else 1')
        
        y = x - mn
    
    y = fold(y, m)
    
    return [y, mn]

def vecadd(x,v):
    """docstring for vecadd"""
    [m,n,o] = x.shape
    x = unfold(x)
    
    [mm,nn] = x.shape
    if v.size == 1:
        x += v
    elif v.shape[0] == 1:
        if v.shape[1] != nn:
            raise Exception('V should have same number of columns as X')
        x += v
    elif v.shape[1] == 1:
        if v.shape[0] != mm:
            raise Exception("V should have same number of rows as X")
        x += v
    
    return fold(x,m)

def vecmult(x,v):
    """docstring for vecmult"""
    [m,n,o] = x.shape
    x = unfold(x)
    
    [mm,nn] = x.shape
    [mv,nv] = v.shape
    
    if mv == mm:
        if nv == nn:
            x *= v
        elif nv == 1:
            x *= v
        else:
            raise Exception('V should be a row vector')
    elif nv == nn:
        if mv == mm:
            x *= v
        elif mv == 1:
            x *= v
        else:
            raise Exception('V should be a column vector')
    else:
        raise Exception('V and X should have the same number of rows or columns')
    
    return fold(x,m)

def normcol(x, w=[]):
    """docstring for normcol"""
    if x.ndim == 3:
        [m,n,o] = x.shape
        x = unfold(x)
        if w == []:
            y = normcol(x)
            y = fold(y,m)
        else:
            if w.shape[0] != m:
                raise Exception('weight matrix should have same ncols as data')
            if w.ndim == 2 and w.shape[1] == 1:
                w = tile(w, (1,m,o))
            if w.shape != x.shape:
                raise Exception('weight should have same size as data')
            w = unfold(w)
            y = normcol(x, w)
            y = fold(y,m)
    else:
        [m,n] = x.shape
        if w == []:
            y = x * ((sum(x ** 2) / m) ** -0.5)
        else:
            if w.shape[0] != x.shape[0]:
                raise Exception('weight matrix should have same ncols as data')
            if w.ndim == 2 and w.shape[1] == 1:
                w = tile(w, (1, n))
            if w.shape != x.shape:
                raise Exception('weight should have same size as data')
            if w.shape[1] == 1:
                w = tile(w,(1,n))
            y = x * (sum((x ** 2) * w) / sum(w)) ** -0.5
        
    return y

def regcov(cxy,cyy,keep=[],threshold=[]):
    """docstring for regcov"""
    
    # PCA of regressor
    [topcs, eigenvalues] = pcarot(cyy)
    
    # discard negligible regressor PCs
    if keep:
        keep = max(keep, topcs.shape[1])
        topcs = topcs[:,0:keep]
        eigenvalues = eigenvalues[0:keep]
    
    if threshold:
        idx = where(eigenvalues/max(eigenvalues) > threshold)
        topcs = topcs[:, idx]
        eigenvalues = eigenvalues[idx]
    
    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = topcs.T * cxy
    
    # projection matrix from regressor PCs
    r = (r * 1/eigenvalues.T)
    
    #projection matrix from regressors
    r = topcs * r
    
    return r

def tsregress(x, y, shifts = [0], keep = [], threshold = [], toobig1 = [], toobig2 = []):
    """docstring for tsregress"""
    
    # shifts must be non-negative
    mn = min(shifts)
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
        
        for k in arange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+mm]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z[0:mm, :] = z[0:mm, :] + y[idx1, :] * r[idx2, :]
        
        z = fold(z,Mx)
        z = z[0:-max(shifts), :, :]
    else:
        [m,n] = x.shape
        z = zeros((m-max(shifts), n))
        for k in arange(nshifts):
            kk = shifts(k)
            idx1 = r_[kk+1:kk+z.shape[0]]
            idx2 = k + r_[0:y.shape[1]] * nshifts
            z = z + y[idx1, :] * r[idx2, :]
    
    offset = max(0, -mn)
    idx = r_[offset+1:offset+z.shape[0]]
    
    return [z, idx]

def sns1(x, nneighbors, skip):
    """docstring for sns1"""
    if x.ndim > 2:
        raise Exception('SNS1 works only with 2D matrices')
    
    [m,n] = x.shape
    
    if not nneighbors:
        nneighbors = n-1
    
    if not skip:
        skip = 0
        
    mn = mean(x)
    x = (x - mn) # remove mean
    N = sqrt(sum(x ** 2))
    NN = 1 / N
    NN[where(isnan(NN))] = 0
    x = (x * NN) # normalize
    
    y = zeros(x.shape)
    
    for k in arange(n):
        
        c1 = x.T * x[:, k]                  #correlation with neighbors
        c1 = c1 / c1[k]                     
        c1[k] = 0                           # demote self
        [c1, idx] = sort(c1 ** 2, 0)[::-1]  # sort
        idx = idx[1+skip:nneighbors+skip]   # keep best
        
        # pca neighbors to orthogonalize them
        xx = x[:,idx]
        c2 = xx.T * xx
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * diag(1/sqrt(eigenvalues))
        
        y[:,k] = tsregress(x[:,k], xx * topcs)
        
        #if mod(k,1000) == 0:
            #[k 100 * sum(y[:,0:k] ** 2) / sum(x[:, 0:k] ** 2)]
        
    y = (y * N)
    
    return y

def find_outliers(x, toobig1, toobig2 = []):
    """docstring for find_outliers"""
    
    [m, n, o] = x.shape
    x = unfold(x)
    
    # remove mean
    x = demean(x)
    
    # apply absolute threshold
    w = ones(x.shape)
    if toobig1:
        w[where(abs(x) > toobig1)] = 0
        x = demean(x,w)
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x,w)
        
        w[where(abs(x) > toobig1)] = 0
        x = demean(x,w)
    else:
        w = ones(x.shape)
        
    # apply relative threshold
    if toobig2:
        X = wmean(x ** 2, w)
        X = tile(X, (x.shape[0], 1))
        idx = where(x**2 > (X * toobig2))
        w[idx] = 0
        
    w = fold(w,m)
    
    return w

def sns0(c, nneighbors, skip=0, wc=[]):
    """docstring for sns0"""
    
    if not wc:
        wc = c
    
    n = c.shape[0]
    
    if not nneighbors:
        nneighbors = n - 1
    
    r = zeros(c.size)
    
    # normalize
    d = sqrt(1 / diag(c))
    c = c * d * d.T
    
    for k in arange(n):
        
        c1 = c[:,k] # correlation of channel
        [c1, idx] = sort(c1 ** 2, 0)[::-1] # sort by correlation
        idx = idx[skip+2:skip+1+nneighbors] # keep best
        
        # pca neighbors to orthogonalize them
        c2 = wc[idx,idx]
        [topcs, eigenvalues] = pcarot(c2)
        topcs = topcs * diag(1/sqrt(eigenvalues))
        
        # augment rotation matric to include this channel
        topcs = array([[1, zeros(nneighbors)], [zeros((nneighbors, 1)), topcs]])
        
        # correlation matrix for rotated data
        c3 = topcs.T * wc[c_[k,idx],c_[k,idx]] * topcs
        
        # first row defines projection to clean component k
        c4 = c3[0,1:] * topcs[1:,1:].T
        
        # insert new column into denoising matric
        r[idx,k] = c4
        
    return c4

def tsxcov(x,y,shifts=0,w=[]):
    """docstring for tsxcov"""
    
    shifts = shifts[:]
    nshifts = shifts.size
    
    [mx,nx,ox] = x.size
    [my,ny,oy] = y.size
    c = zeros((nx, ny*nshifts))
    
    if w:
        x = fold(unfold(x) * unfold(w), mx)
        
    # cross covariance
    for k in arange(ox):
        yy = multishift(y[:,:,k], shifts)
        xx = x[1:yy.shape[0],:,k]
        c = c + xx.T * yy
    
    if not w:
        tw = ox * ny * yy.shape[0]
    else:
        w = w[1:yy.shape[0],:,:]
        tw = sum(w[:])
        
    return [c, tw]

def wmean(x,w=[],dim=0):
    """docstring for wmean"""
    
    if not w:
        y = mean(x,dim)
    else:
        if x.shape[0] != w.shape[0]:
            raise Exception 'data and weight must have same nrows'
        if w.shape[1] == 1:
            w = tile(w,(1,x.shape(1)))
        if w.shape[1] != x.shape[1]:
            raise Exception 'weight must have same ncols as data, or 1'
        y = sum(x * w, dim) / sum(w,dim)
    
    return y

def tsr(x,ref,shifts=0,wx=[],wref=[],keep=[],thresh=10**-20):
    """docstring for tsr"""
    
    #adjust to make shifts non-negative
    n0 = x.shape[0]
    offset1 = max(0,-min(shifts))
    idx = r_[ offset1:x.shape[0] ]
    x = x[idx,:,:]
    if wx:
        wx = wx[idx,:,:]
    ref = ref[0:-offset1,:,:]
    if wref:
        wref = wref[0:-offset1,:,:]
    shifts += offset1
    
    # adjust size of x
    offset2 = max(0,(max(shifts)))
    idx = r_[0:x.shape[0]-offset2]
    x = x[idx,:,:]
    if wx:
        wx = wx[idx,:,:]
        
    [mx,nx,ox] = x.shape
    [mref,nref,oref] = ref.shape
    
    # consolidate weights into single weight matrix
    w = zeros((mx, 1, oref))
    if not wx and not wref:
        w[0:mx,:,:] = 1
    elif not wref:
        w[:,:,:] = wx[:,:,:]
    elif not wx:
        for k in arange(ox):
            wr = wref[:,:,k]
            wr = multishift(wr, shifts)
            wr = wr.min(1)
            w[:,:,k] = wr
    else:
        for k in arange(ox):
            wr = wref[:,:,k]
            wr = multishift(wr, shifts)
            wr = wr.min(1)
            wr = min(wr, wx[0:wr.shape[0],:,k])
            w[:,:,k] = wr
    
    wx = w
    wref = zeros((mref,1,oref))
    wref[idx,:,:] = w
    
    # remove weighted means
    [x, mn1] = demean(x,wx)
    ref = demean(ref,wref)[0]
    
    # equalize power of ref channels, the equalize power of the ref PCs
    ref = normcol(ref,wref)
    ref = tspca(ref,0,[], 10 ** -6)
    ref = normcol(ref,wref)
    
    #covariances and cros covariance with time-shifted refs
    [cref, twcref] = tscov(ref, shifts, wref)
    [cxref, twcxref] = tsxcov(x,ref,shifts,wx)
    
    # regression matrix of x on time-shifted refs
    r = regcov(cxref/twcxref, cref/twcref, keep, thresh)
    
    # TSPCA: clean x by removing regression on time-shifted refs
    y = zeros((mx,nx,ox))
    for k in arange(ox):
        z = multishift(ref[:,:,k], shifts) * r
        y[:,:,k] = x[0:z.shape[0],:,k] - z
    
    [y,mn2] = demean(y,wx)
    
    idx = r_[1+offset1:n0-offset2]
    mn = mn1 + mn2
    w = wref
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    