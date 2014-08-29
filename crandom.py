__author__ = 'Till Hoffmann'

import numpy as np

def categorical(pvals, size=None):
    """
    Draw samples from a categorical distribution.

    Parameters
    ----------
    pvals : sequence of floats, length p or array of floats, shape (p, ...)
        Probabilities of each of the ``p`` different outcomes. The entries are
        renormalised such that an unnormalised distribution can be provided.
    size : tuple of ints
        Given a `size` of ``(M, N, K)``, then ``M*N*K`` samples are drawn,
        and the output shape becomes ``(..., M, N, K)``, since each sample
        has shape ``(...)``.
    """
    #Convert to a numpy array
    if type(pvals) is not np.array:
        pvals = np.array(pvals)

    #Make sure the size is a tuple
    if size is None or size == 1:
        size = tuple()
    elif type(size) is int:
        size = (size,)
    else:
        assert type(size) is tuple

    #Compute the cumulative distribution
    cvals = np.cumsum(pvals, axis=0)
    for _ in range(len(size)):
        cvals = np.expand_dims(cvals, cvals.ndim)
    #Generate uniform random variables
    x = np.random.uniform(size = cvals.shape[1:] + size) * cvals[-1]
    #Get the categories
    x = np.argmin(cvals <= x, axis = 0)
    return x


if __name__ == '__main__':
    pvals = np.linspace(1, 4, 4)
    x = categorical(pvals, 1000)
    print np.bincount(x), pvals / np.sum(pvals)

