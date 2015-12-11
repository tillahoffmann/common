import numpy as np
import inspect
from matplotlib.colors import LinearSegmentedColormap, ColorConverter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def credible_interval(samples, summary='mean', tail=0.5):
    """
    Returns the summary statistic together with the credible interval
    defined by the tail probability in percent.
    """
    # Obtain the summary statistic
    if summary=='mean':
        s = np.mean(samples, axis=0)
    elif summary=='median':
        s = np.median(samples, axis=0)
    elif callable(summary):
        s = summary(samples)
    else:
        raise NotImplementedError("Summary statistic `{}` is not recognised "
                                  "and is not callable.".format(summary))
        
    # Get the tail probabilities
    if hasattr(tail, '__iter__'):
        lower_tail, upper_tail = tail
    else:
        lower_tail = upper_tail = tail
        
    # Obtain the credible bounds
    lower = np.percentile(samples, lower_tail, axis=0)
    upper = np.percentile(samples, 100 - upper_tail, axis=0)
    return np.asarray((s, lower, upper))


class Namespace:
    """
    A container for values that can be accessed as member variables.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @staticmethod
    def from_arguments(function, _locals):
        """
        Creates a namespace from the arguments passed to
        `function` using the values in `_locals`.
        """
        return Namespace(**{arg: _locals[arg] for arg in
                            inspect.getargspec(function).args})


def build_color_map(values, colors):
    """
    Build a `LinearSegmentedColormap`.
    
    Parameters
    ----------
    values : array
        array whose entries correspond to colors
    colors : array
        array whose entries correspond to values
    """
    # Convert all colors to RGB
    colors = ColorConverter().to_rgba_array(colors)
    # Normalise the values
    values = 1.0 * np.asarray(values) / np.max(values)
    # Build a dictionary
    cdict = {color : zip(values, colors[:, i], colors[:, i])
        for i, color in enumerate(['red', 'green', 'blue'])}
    return LinearSegmentedColormap('temp', cdict)


def extended_range(x, num=None, f=0.1):
    """
    Evaluate an extended range of an array.

    Parameters
    ----------
    x : array
        array to compute a range for
    num : int, optional
        number of samples to generate. If `None`, the extended interval will be returned.
    f : float
        fraction by which to extend the range above and below

    Returns
    -------
        evenly spaced numbers over the extended range if `num` is given; extended interval otherwise
    """
    xmin = np.min(x)
    xmax = np.max(x)
    xrng = xmax - xmin
    xmin -= f * xrng
    xmax += f * xrng
    if num is not None:
        return np.linspace(xmin, xmax, num)
    else:
        return xmin, xmax


def parameter_density_plot(samples, names=None, num=100, burnin=None, start=None, stop=None, nrows=None, ncols=None,
                           fig=None):
    # Try to load the samples from file if possible
    if type(samples) is str:
        samples = np.load(samples)
    n, p = samples.shape
    # Determine the starting and stopping indices
    start = start or 0
    stop = stop or p
    count = stop - start
    # Determine the number of rows and columns
    aspect = 4.0 / 3
    ncols = int(ncols or np.ceil(np.sqrt(count / aspect)))
    nrows = int(nrows or np.ceil(count / float(ncols)))

    # Use half the samples as burnin by default
    burnin = burnin or n / 2

    # Use default names
    names = names or ['param{}'.format(i) for i in range(start, stop)]

    fig = fig or plt.gcf()

    # Iterate over the parameters
    for i in range(start, stop):
        x = samples[burnin:, i]
        name = names[i - start]
        # Print a summary of the variable
        print name
        print "=" * len(name)
        print "\tMean: {}".format(np.mean(x))
        print "\tStd: {}".format(np.std(x))
        print "\tMedian: {}".format(np.median(x))
        print "\tIQR: {}".format(np.percentile(x,75) - np.percentile(x,25))
        print

        # Plot a kernel density estimate and a histogram
        ax = fig.add_subplot(nrows, ncols, 1 + i - start)
        ax.hist(x, histtype='stepfilled', color='silver', normed=True)
        kde = gaussian_kde(x)
        linx = extended_range(x, num)
        ax.plot(linx, kde(linx))
        ax.set_title(name)

    fig.tight_layout()

    # Print the parameter covariance matrix
    print "Covariance"
    print "=========="
    print np.cov(samples[burnin:, start:stop].T)

    return fig

