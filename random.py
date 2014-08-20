__author__ = 'Till Hoffmann'

import numpy as np

def categorical(pvals):
    """
    Draw samples from a categorical distribution.

    Parameters
    ----------
    pvals : sequence of floats, length p or array of floats, length n*p
        Probabilities of each of the ``p`` different outcomes. The entries are
        renormalised such that an unnormalised distribution can be provided.

        ``n`` rows of probabilities of each of the ``p`` different outcomes.
        The entries are renormalised such that an unnormalised distribution
        can be provided.
    """
    #Convert to a numpy array
    if type(pvals) is not np.array:
        pvals = np.array(pvals)

    #Reshape to dimension 2 if it is of dimension one
    if pvals.ndim == 1:
        pvals = np.reshape(pvals, (1, len(pvals)))

    #Compute the cumulative distribution
    cvals = np.cumsum(pvals, axis=1)
    #Generate uniform random variables
    x = np.random.uniform(size = len(cvals)) * cvals[:,-1]
    #Get the categories
    x = np.argmin(cvals <= x[:,np.newaxis], axis = 1)
    return x


if __name__ == '__main__':
    pvals = np.linspace(1, 4, 4)
    x = categorical(np.vstack((pvals,) * 1000))
    print np.bincount(x), pvals / np.sum(pvals)

