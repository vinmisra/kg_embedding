import numpy as np
from est_entro import est_entro_JVHW

def est_MI_JVHW(X, Y):
    """Return mutual information estimates using JVHW entropy estimator.
    
    This function returns our scalar estimate of the mutual information (in bits)
    I(X;Y) when both X and Y are vectors, and returns a vector consisting
    of the estimate of mutual information between each corresponding column
    of X and Y when they are matrices.
    Input:
    ----- X, Y: two vectors or matrices (in numpy.array type) with the same size,
    which can only contain integers
    Output:
    ----- est: the estimate of the mutual information (in bits) between input vectors
    or that between each corresponding column of the input matrices
    """
    
    if X.shape != Y.shape:
        print('Input arguments X and Y should be of the same size!')
        return
    
    Y_uni, Y_r = np.unique(Y, return_inverse=True)
    Y_r = Y_r.reshape(Y.shape)
    Ny = len(Y_uni)
    
    est = est_entro_JVHW(X) + est_entro_JVHW(Y_r) - est_entro_JVHW(X*Ny+Y_r)
    return np.maximum(est, 0)