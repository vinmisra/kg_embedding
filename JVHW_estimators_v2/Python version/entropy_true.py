import numpy as np

def entropy_true(x):
    """Compute the Shannon entropy of a discrete distribution x in bits."""
    x = np.array(x)
    non_zero = x >= 1e-10
    output = -x[non_zero] * np.log(x[non_zero])
    return sum(output) / np.log(2)