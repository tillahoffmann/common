import numpy as np

class Gibbs(object):
    """
    Abstract base class for simple Gibbs sampling with conjugate priors.

    Parameters
    ----------
    *prior_params : array_like
        Parameters that characterise the prior.

    Attributes
    ----------
    prior_params : array_like
        Parameters that characterise the prior.
    posterior_params : array_like
        Parameters that characterise the posterior.
    """
    def __init__(self, *prior_params):
        self.prior_params = np.asarray(prior_params)
        self.posterior_params = None
        
    def evaluate_posterior_params(self, X, weight=None):
        """
        Evaluate the posterior parameters.

        Parameters
        ----------
        X : array_like
            The data to infer parameters from.
        weight : array_like, float, optional
            Weights associated with individual data points.
        """
        raise NotImplementedError("Inheriting classes should implement this "
                                  "function to compute posterior parameters "
                                  "from the prior parameters and data.")
        
    def sample(self, params, size=None):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        params : array_like
            Parameters that characterise the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        raise NotImplementedError("Inheriting classes should implement this "
                                  "function to draw a sample from the "
                                  "distribution with the given parameters.")
        
    def sample_prior(self, size=None):
        """
        Draw samples from the prior.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        return self.sample(self.prior_params, size)
    
    def sample_posterior(self, size=None):
        """
        Draw samples from the posterior.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        # Ensure posterior parameters are available
        if self.posterior_params is None:
            raise RuntimeError("Posterior parameters are not defined. Call "
                               "`evaluate_posterior_params` first.")
        return self.sample(self.posterior_params, size)
    
    @staticmethod
    def evaluate_likelihood(Y, sample):
        """
        Evaluate the conjugate likelihood.
        
        Parameters
        ----------
        Y : array_like
            The values at which to evaluate the likelihood.
        sample : array_like
            A sample from the distribution characterising the likelihood.
        """
        raise NotImplementedError("Inheriting classes should implement this "
                                  "function to evaluate the likelihood that "
                                  "the distribution is conjugate to.")

class NormalGamma(Gibbs):
    """
    A normal-gamma distribution which is a conjugate prior for normally-
    distributed data with unknown mean and precision.

    Parameters
    ----------
    mu : float
        Prior mean of the normal distribution.
    lmbda : float >= 0
        Prior precision of the normal distribution. `lmbda=0` corresponds
        to a vague but improper prior.
    alpha : float > 0
        Prior shape parameter of the gamma distribution. `alpha=0` corresponds
        to an exponential distribution.
    beta : float >= 0
        Prior decay rate of the gamma distribution. `beta=0` corresopnds
        to a vague but improper prior.
    """
    def __init__(self, mu, lmbda, alpha, beta):
        super(NormalGamma, self).__init__(mu, lmbda, alpha, beta)
        
    def sample(self, params, size=None):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        params : array_like
            Parameters that characterise the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        mu, lmbda, alpha, beta = params
        # Draw samples from the gamma distribution
        samples_gamma = np.random.gamma(alpha, 1.0 / beta, size)
        # Compute the std for the normal component
        std = 1.0 / np.sqrt(lmbda * samples_gamma)
        samples_normal = mu + std * np.random.normal(size=size)
        # Return the samples
        return samples_normal, samples_gamma
        
    def evaluate_posterior_params(self, X, weight=None):
        """
        Evaluate the posterior parameters.

        Parameters
        ----------
        X : array_like
            The data to infer parameters from.
        weight : array_like, float, optional
            Weights associated with individual data points.
        """
        # Convert to an array
        X = np.asarray(X)
        # Choose default weights if they are not available
        if weight is None:
            weight = np.ones(X.shape[0])
        # Unpack parameters
        mu, lmbda, alpha, beta = self.prior_params
        # Compute the total weight
        total = np.sum(weight)
        # Compute the mean and standard deviation
        mean = np.sum(weight * X) / total
        var = np.sum(weight * (X - mean) ** 2) / total
        # Update the parameter values
        lmbda2 = lmbda + total
        mu2 = lmbda * mu + total * mean / (lmbda2)
        alpha2 = alpha + total / 2
        beta2 = beta + 0.5 * (total * var + lmbda * total 
                              * (mean - mu) ** 2 / lmbda2)
        # Store the parameters
        self.posterior_params = np.asarray([mu2, lmbda2, alpha2, beta2])
    
    @staticmethod
    def evaluate_likelihood(Y, sample):
        """
        Evaluate the conjugate likelihood which is a normal distribution.
        
        Parameters
        ----------
        Y : array_like
            The values at which to evaluate the likelihood.
        sample : array_like
            A sample of the mean and precision.
        """
        # Extract the mean and precision
        mean, precision = np.asarray(sample)
        # Compute the distribution
        chi2 = (Y[:, None] - mean) ** 2 * precision
        pdf = (np.sqrt(precision / (2 * np.pi)) * np.exp(-0.5 * chi2)).T
        # Return the first entry if there is only one sample
        if pdf.shape[0] == 1:
            return pdf[0]
        else:
            return pdf
        
class NormalInverseGamma(NormalGamma):
    """
    A normal-gamma distribution which is a conjugate prior for normally-
    distributed data with unknown mean and variance.

    Parameters
    ----------
    mu : float
        Prior mean of the normal distribution.
    Sigma : float > 0
        Prior variance of the normal distribution. `lmbda=np.inf` corresponds
        to a vague but improper prior.
    alpha : float > 0
        Prior shape parameter of the gamma distribution. `alpha=0` corresponds
        to an exponential distribution.
    beta : float >= 0
        Prior decay rate of the gamma distribution. `beta=0` corresopnds
        to a vague but improper prior.
    """
    def __init__(self, mu, Sigma, alpha, beta):
        super(NormalInverseGamma, self).__init__(mu, 1.0 / Sigma, alpha, beta)
        
    def sample(self, params, size=None):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        params : array_like
            Parameters that characterise the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        """
        samples_mean, samples_precision = super(NormalInverseGamma, self).sample(params, size)
        return samples_mean, 1.0 / samples_precision
    
    @staticmethod
    def evaluate_likelihood(Y, sample):
        """
        Evaluate the conjugate likelihood which is a normal distribution.
        
        Parameters
        ----------
        Y : array_like
            The values at which to evaluate the likelihood.
        sample : array_like
            A sample of the mean and variance.
        """
        return NormalGamma.evaluate_likelihood(Y, (sample[0], 1.0 / sample[1]))