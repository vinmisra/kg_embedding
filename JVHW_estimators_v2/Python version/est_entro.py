import numpy as np
import scipy.io as sio
from math import log
from scipy.sparse import csr_matrix

def est_entro_JVHW(samp):
    """Return entropy estimates using JVHW estimator.
    
    This function returns our scalar estimate of the entropy (in bits) of samp
    when samp is a vector, and returns a row vector consisting of the entropy
    estimate of each column of samp when samp is a matrix.
    Input: 
    ----- samp: a vector or matrix (in numpy.array type) which can only contain integers
    Output: 
    ----- est: the entropy (in bits) of the input vector or that of each column
    of the input matrix
    """
    
    if samp.ndim == 2:
        n, wid = samp.shape
        if n == 1:
            samp = samp.transpose()
            n = wid
            wid = 1
    elif samp.ndim == 1:
        samp = np.array([samp])
        samp = samp.transpose()
        n = samp.size
        wid = 1
    else:
        print('The input "samp" is not a vector or a 2D matrix!')
        return
    
    V = [[0.3303, -0.3295, 0.4679]]
    order = 4 + np.ceil(1.2 * log(n))
    mat_contents = sio.loadmat('poly_coeff.mat')
    coeff = mat_contents['poly_coeff'][order-1, 0][0]
    
    for i in range(wid):
        samp[:, i] = np.unique(samp[:, i], return_inverse=True)[1]
    
    fingerprint = int_hist(int_hist(samp))[1:]
    len_f, wid_f = fingerprint.shape
    est = np.zeros(wid_f)
    
    if len_f > 0:
        prob = np.linspace(1.0/n, len_f*1.0/n, len_f)
        f1nonzero = fingerprint[0] > 0
        lenf1nonzero = list(f1nonzero).count(True)
        c_1 = np.zeros(wid_f)
        if n > 15 and lenf1nonzero > 0:
            c_1[f1nonzero] = np.dot(V, \
                np.array((log(n)*np.ones(lenf1nonzero), np.log(fingerprint[0, f1nonzero]), np.ones(lenf1nonzero))))
            c_1 = np.maximum(c_1, 1/(1.9*log(n)))
        
        prob_mat = entro_mat(prob, n, coeff, c_1)
        for i in range(wid_f):
            est[i] = np.dot(fingerprint[:, i], prob_mat[:, i]) / log(2)
    return est

def est_entro_MLE(samp):
    """Return entropy estimates using maximum likelihood estimatation (MLE).
    
    This function returns our scalar estimate of the entropy (in bits) of samp
    when samp is a vector, and returns a row vector consisting of the entropy
    estimate of each column of samp when samp is a matrix.
    Input: 
    ----- samp: a vector or matrix (in numpy.array type) which can only contain integers
    Output: 
    ----- est: the entropy (in bits) of the input vector or that of each column
    of the input matrix
    """
    
    if samp.ndim == 2:
        n, wid = samp.shape
        if n == 1:
            samp = samp.transpose()
            n = wid
            wid = 1
    elif samp.ndim == 1:
        samp = np.array([samp])
        samp = samp.transpose()
        n = samp.size
        wid = 1
    else:
        print('The input "samp" is not a vector or a 2D matrix!')
        return
    
    for i in range(wid):
        samp[:, i] = np.unique(samp[:, i], return_inverse=True)[1]
    
    fingerprint = int_hist(int_hist(samp))[1:]
    len_f, wid_f = fingerprint.shape
    est = np.zeros(wid_f)
    if len_f > 0:
        prob = np.linspace(1.0/n, len_f*1.0/n, len_f)
        prob_mat = xlogx(prob)
        est = np.dot(prob_mat, fingerprint) / log(2)
    return est

def xlogx(x):
    x = np.array(x)
    non_zero = x >= 1e-10
    output = np.zeros(len(x))
    output[non_zero] = -x[non_zero] * np.log(x[non_zero])
    return output

def entro_mat(x, n, g_coeff, c_1):
    len_x = len(x)
    order = len(g_coeff)
    thres = 4 * c_1 * log(n)/n
    output = np.zeros((len_x, len(c_1)))
    for j in range(len_x):
        value = np.ones((order, len(c_1)))
        region_nonlarge = x[j] <= thres
        region_mid = np.logical_and(region_nonlarge, 2*x[j] > thres)
        if any(region_nonlarge):
            for q in range(1, order):
                value[-q-1, region_nonlarge] = value[-q, region_nonlarge] * (n*x[j] - q + 1) \
                    / (n - q + 1) / thres[region_nonlarge]
            output[j, region_nonlarge] = (np.dot(g_coeff, value[:, region_nonlarge]) \
                - value[-2, region_nonlarge] * np.log(thres[region_nonlarge])) * thres[region_nonlarge]
        output[j, ~region_nonlarge] = -x[j]*log(x[j]) + 1.0/(2*n)
        if any(region_mid):
            ratio = 2*x[j]/thres[region_mid] - 1
            output[j, region_mid] = ratio * (-x[j]*log(x[j]) + 1.0/(2*n)) + (1-ratio) * output[j, region_mid]
    output = np.maximum(output, 0)
    return output


def int_hist(x):
    """Return the histogram of all integer values 0 : x.max()"""
    n, wid_x = x.shape
    large = x.max()
    h = np.zeros((large+1, wid_x))
    for i in range(wid_x):
        row = np.arange(n)
        col = x[:, i]
        data = np.ones(n)
        h[:, i] = np.sum(csr_matrix((data, (row, col)), shape=(n, large+1)).toarray(), 0)
    return h