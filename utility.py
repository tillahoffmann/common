import numpy as np
import inspect

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
