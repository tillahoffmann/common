import numpy as np
import types


def prior_posterior(obj):
    """
    Create prior and posterior convenience functions.

    Parameters
    ----------
    obj : class or function
        A function for which to create prior and posterior shortcuts or a class for
        whose functions shortcuts should be created.
    """
    def build_functions(name):
        """
        Build prior and posterior wrappers for a function.

        Parameters
        ----------
        name : Name of the function.
        """
        def prior(self, *args, **kwargs):
            # Get the function by name to support inheritance
            _func = self.__class__.__dict__[name]
            # Call the function
            return _func(self, self.prior_params, *args, **kwargs)

        def posterior(self, *args, **kwargs):
            # Ensure posterior parameters are set
            if self.posterior_params is None:
                raise RuntimeError("Posterior parameters must be set first.")
            # Get the function by name to support inheritance
            _func = self.__class__.__dict__[name]
            # Call the function
            return _func(self, self.posterior_params, *args, **kwargs)

        # Set the docstrings
        prior.__doc__ = posterior.__doc__ = 'See `{}` for details.'.format(name)
        return prior, posterior

    if isinstance(obj, types.ClassType):
        # Iterate over all functions
        for name, func in obj.__dict__.items():
            # Check whether they have the decorator
            if hasattr(func, 'prior_posterior'):
                # Set the prior and posterior functions
                obj.__dict__['prior_' + name], obj.__dict__['posterior_' + name] = build_functions(name)
                # Delete the attribute
                del func.prior_posterior
    elif isinstance(obj, types.FunctionType):
        # Set the flag for the function
        obj.prior_posterior = None

    return obj


@prior_posterior
class Conjugate:
    """
    Abstract base class for simple Gibbs sampling with conjugate priors.

    Parameters
    ----------
    *prior_params : list
        Parameters that characterise the prior.

    Attributes
    ----------
    prior_params : list
        Parameters that characterise the prior.
    posterior_params : list
        Parameters that characterise the posterior.
    """
    def __init__(self, prior_params, copy_prior=False):
        self.prior_params = prior_params
        self.posterior_params = list(prior_params) if copy_prior else None


    def __len__(self):
        # Ensure the prior parameters are given
        if len(self.prior_params) == 0:
            raise NotImplementedError
        # Get the first parameter
        first = self.prior_params[0]
        # Check whether it supports the 'length' attribute
        if not hasattr(first, '__len__'):
            raise NotImplementedError
        # Return the length of the first item
        return len(first)

    @prior_posterior
    def sample(self, params, size=None):
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        raise NotImplementedError

    @prior_posterior
    def mean(self, params):
        """
        Evaluate the mean of the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        """
        raise NotImplementedError

    @prior_posterior
    def covariance(self, params):
        """
        Evaluate the (co)variance of the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        """
        raise NotImplementedError

    @prior_posterior
    def precision(self, params):
        """
        Evaluate the precision of the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        """
        raise NotImplementedError

    @prior_posterior
    def square(self, params):
        """
        Evaluate second moment of the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        """
        raise NotImplementedError


class Normal(Conjugate):
    def __init__(self, loc=0.0, precision=0.0, copy_prior=False):
        Conjugate.__init__(self, (loc, precision), copy_prior)

    def sample(self, params, size=None):
        loc, precision = params
        return np.random.normal(loc, 1.0 / np.sqrt(precision), size)

    def mean(self, params):
        return params[0]

    def covariance(self, params):
        return 1.0 / params[1]

    def precision(self, params):
        return params[1]

    def square(self, params):
        loc, precision = params
        return loc * loc + 1.0 / precision

    def update_normal_mean(self, data=None, generative_precision=None, data_precision=None, data_total=None):
        """
        Update the posterior parameters assuming the distribution is a conjugate prior for the population mean
        of normally-distributed data with known generative precision.

        Parameters
        ----------
        data : list
            normally-distributed data with known generative precision
        generative_precision : float
            precision of the process that generated the data
        data_precision : float
            precision that results from the observed data (calculated if data are given)
        data_total : float
            sum of all data (calculated if data are given)

        Notes
        -----
        The `data_precision` and `data_total` parameters are calculated automatically if data are given. However, in
        complex situations it may be advantageous to provide the parameters explicitly.
        """
        # Get the prior parameters and take the mean if necessary
        prior_loc, prior_precision = self.prior_params
        if isinstance(prior_loc, Conjugate):
            prior_loc = prior_loc.posterior_mean()
        if isinstance(prior_precision, Conjugate):
            prior_precision = prior_precision.posterior_mean()

        # Calculate the update parameters from the data
        if data is not None:
            # Take the mean if the generative precision is uncertain
            if isinstance(generative_precision, Conjugate):
                generative_precision = generative_precision.posterior_mean()
            # Same for the data
            if isinstance(data, Conjugate):
                data = data.posterior_mean()
            data_precision = generative_precision * len(data)
            data_total = generative_precision * np.sum(data, axis=0)
        # Or ensure that they were given
        elif data_precision is None or data_total is None:
            raise ValueError("Either `data` and `generative_precision` or `data_precision` "
                             "and `data_total` must be specified.")

        # Update the posterior
        posterior_precision = prior_precision + data_precision
        posterior_loc = (prior_precision * prior_loc + data_total) / posterior_precision
        self.posterior_params = [posterior_loc, posterior_precision]
        return self.posterior_params


@prior_posterior
class Multinormal(Conjugate):
    def sample(self, params, size=None):
        loc = params[0]
        cov = self.covariance(params)
        return np.random.normal(loc, cov, size)

    def mean(self, params):
        return params[0]

    def covariance(self, params):
        return params[2] if len(params) == 3 else np.linalg.inv(params[1])

    def precision(self, params):
        return params[1]

    def square(self, params):
        loc = params[0]
        cov = self.covariance(params)
        return loc * loc + np.diag(cov)

    @prior_posterior
    def outer(self, params):
        """
        Evaluate the expected outer product of the distribution.

        Parameters
        ----------
        params : list
            Parameters that characterise the distribution.
        """
        loc = params[0]
        cov = self.covariance(params)
        return loc[:, None] * loc[None, :] + cov


class Gamma(Conjugate):
    def __init__(self, shape=1e-3, scale=1e-3, copy_prior=False):
        Conjugate.__init__(self, (shape, scale), copy_prior)

    def sample(self, params, size=None):
        a, b = params
        return np.random.gamma(a, 1.0 / b, size)

    def mean(self, params):
        a, b = params
        return 1.0 * a / b

    def covariance(self, params):
        a, b = params
        return 1.0 * a / (b * b)

    def precision(self, params):
        a, b = params
        return 1.0 * b * b / a

    def square(self, params):
        a, b = params
        return 1.0 * a * (a - 1) / (b * b)

    def update_normal_precision(self, data=None, generative_mean=None, data_shape=None, data_scale=None):
        """
        Update the posterior parameters assuming the distribution is a conjugate prior for the population precision
        of normally-distributed data with known generative mean.

        Parameters
        ----------
        data : list
            normally-distributed data with known generative mean
        generative_mean : float
            mean of the process that generated the data
        data_shape : float
            shape that results from the observed data (calculated if data are given)
        data_scale : float
            scale that results from the observed data (calculated if data are given)

        Notes
        -----
        The `data_shape` and `data_scale` parameters are calculated automatically if data are given. However, in
        complex situations it may be advantageous to provide the parameters explicitly.
        """
        # Get the prior parameters and take the mean if necessary
        prior_shape, prior_scale = self.prior_params
        if isinstance(prior_shape, Conjugate):
            prior_shape = prior_shape.posterior_mean()
        if isinstance(prior_scale, Conjugate):
            prior_scale = prior_scale.posterior_mean()

        # Calculate the update parameters from the data
        if data is not None:
            data_shape = 0.5 * len(data)
            # Check if the data are uncertain themselves and take the mean if so
            if isinstance(data, Conjugate):
                data_square = data.posterior_square()
                data = data.posterior_mean()
            else:
                data_square = data * data
            # Check if the generative mean is uncertain and take the mean if so
            if isinstance(generative_mean, Conjugate):
                generative_mean_square = generative_mean.posterior_square()
                generative_mean = generative_mean.posterior_mean()
            else:
                generative_mean_square = generative_mean * generative_mean

            data_scale = 0.5 * np.sum(data_square - 2 * data * generative_mean + generative_mean_square)
        # Or ensure that they were given
        elif data_shape is None or data_scale is None:
            raise ValueError("Either `data` and `generative_mean` or `data_shape` "
                             "and `data_scale` must be specified.")

        # Update the posterior
        posterior_shape = prior_shape + data_shape
        posterior_scale = prior_scale + data_scale
        self.posterior_params = [posterior_shape, posterior_scale]
        return self.posterior_params


def __main__():
    import matplotlib.pyplot as plt
    # Generate some random numbers
    generative_mean = 2
    generative_precision = 4
    data = np.random.normal(generative_mean, generative_precision ** -0.5, 100)
    # Create a normal and a gamma random variable (they have non-informative parametrisations by default)
    mean = Normal()
    precision = Gamma()

    # Run a Gibbs sampler starting the precision with the prior mean
    precision_sample = precision.mean_prior()
    steps = 1000
    samples = []
    for step in range(steps):
        # Update the posteriors conditional on the other variable and sample
        mean.update_normal_mean(data, generative_precision=precision_sample)
        mean_sample = mean.posterior_sample()
        precision.update_normal_precision(data, generative_mean=mean_sample)
        precision_sample = precision.posterior_sample()
        samples.append((mean_sample, precision_sample))

    # Run a variational algorithm
    for step in range(steps):
        mean.update_normal_mean(data, precision)
        precision.update_normal_precision(data, mean)

    # Remove some burnin
    samples = np.asarray(samples[500:])

    # Print some summaries
    print "Mean"
    print "===="
    print "Gibbs      : {} +- {}".format(np.mean(samples[:,0]), np.std(samples[:,0]))
    print "Variational: {} +- {}".format(mean.posterior_mean(), np.sqrt(mean.posterior_covariance()))
    print
    print "Precision"
    print "========="
    print "Gibbs      : {} +- {}".format(np.mean(samples[:,1]), np.std(samples[:,1]))
    print "Variational: {} +- {}".format(precision.posterior_mean(), np.sqrt(precision.posterior_covariance()))

    # Plot the results
    _x, _y = np.transpose(samples)
    plt.plot(_x, _y, marker='o')
    plt.axvline(generative_mean, color='r')
    plt.axhline(generative_precision, color='r')
    plt.show()


if __name__=='__main__':
    __main__()