import src

import numpy

SMOOTH_SLOPE = 0.1 # i.e. it takes AD 10 years to go from 0 to 1.

def fit(T, Y, M, **kwargs):

    '''

    Input:
        T - numpy array of shape (N, S), ages.
        Y - numpy array of shape (N, S), diagnoses.
        M - numpy array of shape (N, S), mask for sequence padding.
        bounds - tuple of (list, list), scipy curve_fit bounds.

    Output:
        P - numpy array of shape (N, C), where C is the number
            of parameters for the best sigmoid fit, which is 4.

    '''
    
    N = T.shape[0]
    z = numpy.zeros((N, 1))
    o = numpy.ones((N, 1))

    T = numpy.concatenate([z, T], axis=1)
    Y = numpy.concatenate([z, Y], axis=1)
    M = numpy.concatenate([o, M], axis=1)

    T, Y = _smoothen(T, Y, M)

    return T[:,1:], Y[:,1:], src.models.sigmoid.fit(T, Y, M, **kwargs)

def _smoothen(T, Y, M):
    N, S = T.shape
    assert (N, S) == Y.shape
    Tp, Yp = T.copy(), Y.copy()
    R = _obtain_earliest_AD_diagnosis_index(Y, M)
    for i in range(N):
        if _is_AD(Yp[i]):
            _smoothen_row(Tp[i], Yp[i], R[i])
    return Tp, Yp

def _is_AD(yp):
    return yp.sum() > 0

def _obtain_earliest_AD_diagnosis_index(Y, M):
    '''

    Description:
        Finds the earliest index corresponding to AD diagnosis,
        or sum of mask if the subject is healthy.
        
    '''
    R = (1-Y).astype(numpy.bool) & M.astype(numpy.bool)
    return R.astype(numpy.int64).sum(axis=1)

def _smoothen_row(tp, yp, r):
    t1 = tp[r]
    t0 = t1 - 1.0/SMOOTH_SLOPE
    th = t1 - 0.5/SMOOTH_SLOPE
    b = 1.0 - t1 *SMOOTH_SLOPE
    
    sloped_region_i = (tp > t0) & (tp <= th)
    yp[sloped_region_i] = tp[sloped_region_i]*SMOOTH_SLOPE + b
    
    saturated_region = (tp > th) & (tp < t1)
    yp[saturated_region] = 0.5
