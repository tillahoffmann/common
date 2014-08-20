__author__ = 'Till Hoffmann'

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


def histogram_interp(a, bins=10, range=None, weights=None, density=None, cumulative=False, k=3):
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
       Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
       If `bins` is an int, it defines the number of equal-width
       bins in the given range (10, by default). If `bins` is a sequence,
       it defines the bin edges, including the rightmost edge, allowing
       for non-uniform bin widths.
    range : (float, float), optional
       The lower and upper range of the bins.  If not provided, range
       is simply ``(a.min(), a.max())``.  Values outside the range are
       ignored.
    normed : deprecated; see `density` instead
    weights : array_like, optional
       An array of weights, of the same shape as `a`.  Each value in `a`
       only contributes its associated weight towards the bin count
       (instead of 1).  If `normed` is True, the weights are normalized,
       so that the integral of the density over the range remains 1
    density : bool, optional
       If False, the result will contain the number of samples
       in each bin.  If True, the result is the value of the
       probability *density* function at the bin, normalized such that
       the *integral* over the range is 1. Note that the sum of the
       histogram values will not be equal to 1 unless bins of unity
       width are chosen; it is not a probability *mass* function.
       Overrides the `normed` keyword if given.

    Returns
    -------
    spline : InterpolatedUnivariateSpline
       An interpolated spline of the values of the histogram. See
       `normed` and `weights` for a description of the possible semantics.
    x_min : float
       Smallest bin edge.
    x_max : float
       Largest bin edge.
    """
    # Get the histogram
    y, edges = np.histogram(a, bins=bins, range=range, weights=weights, density=density)
    x = 0.5 * (edges[:-1] + edges[1:])
    if cumulative:
        y = np.cumsum(y)
    #Spline the result
    spline = InterpolatedUnivariateSpline(x, y, k=k)
    #Compute the normalisation if desired
    if density:
        if cumulative:
            normalisation = spline(edges[-1])
        else:
            normalisation = spline.integral(edges[0], edges[-1])
        spline = InterpolatedUnivariateSpline(x, y / normalisation, k=k)
    return spline, edges[0], edges[-1]


def hist_interp(a, bins=10, range=None, weights=None, density=None, cumulative=False, k=3,
                num=50, ax=None, **kwargs):
    spline, x_min, x_max = histogram_interp(a, bins, range, weights, density, cumulative, k)
    x = np.linspace(x_min, x_max, num)
    if ax is None:
        ax = plt.gca()
    return ax.plot(x, spline(x), **kwargs)