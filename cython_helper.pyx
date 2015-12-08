from libc.math cimport sqrt, exp, log
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
import numpy as np

def seed_rng(s):
    """
    Initialise the standard pseud-random number generator.
    """
    if s is None:
        s = time(NULL)
    srand(s)


cdef double uniform():
    """
    Draw a sample from the standard uniform distribution.
    """
    return rand() / float(RAND_MAX)


def truncnorm(double[:] lower, double[:] out=None):
    """
    Generate standard normal random variables with a lower truncation.

    References
    ----------
    C. Robert. Simulation of truncated normal variables. Statistics and Computing. 5(2):121--125, 1995.
    """
    cdef double rate, z, rho, _lower
    cdef int n = lower.shape[0], i
    # Create an array if necessary
    if out is None:
        out = np.empty(n)
    # Iterate over the elements
    for i in range(n):
        # Use the algorithm in Robert to generate random variables
        _lower = lower[i]
        rate = 0.5 * (_lower + sqrt(_lower * _lower + 4))
        while True:
            z = - log(uniform()) / rate + _lower
            rho = exp(-(z-rate) * (z-rate) / 2)
            if uniform() <= rho:
                out[i] = z
                break
    return out