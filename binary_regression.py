import numpy as np
from scipy.special import expit as logistic, erfc, gammaln
from pyximport import install
from warnings import warn
install(setup_args=dict(include_dirs=np.get_include()))
from cython_helper import truncnorm, seed_rng


def normal_cdf(z):
    """
    Evaluate the CDF of the standard normal distribution.
    """
    return 0.5 * erfc(- z / np.sqrt(2))


def normal_pdf(z):
    """
    Evaluate the PDF of the standard normal distribution.
    """
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def inverse_mills_ratio(z, approx=True):
    """
    Evaluate the inverse Mill's ratio.
    """
    z = np.asarray(z, dtype=float)
    # Evaluate the Mills ratio without approximations (breaks down for large |z|)
    if not approx:
        return  normal_pdf(z) / normal_cdf(z)
    else:
        # Use approximation which is correct up to 10^{-10} in relative terms
        return np.piecewise(z, [z < -20, z > 10],
                            [lambda z: -z - 1.0 / z + 2 / z ** 3 - 10 / z ** 5 + 74 / z ** 7,
                             lambda z: 1.0 / (np.exp(z * z / 2) * np.sqrt(2 * np.pi) - 1.0 / z),
                             lambda z: normal_pdf(z) / normal_cdf(z)])


def logistic_lambda(z, logistic_z=None):
    """
    Evaluate the $\\lambda$ function in logistic regression.
    """
    if logistic_z is None:
        logistic_z = logistic(z)
    return np.where(z == 0, .125, (logistic_z - 0.5) / (2 * z))


class BinaryRegression(object):
    def __init__(self, design_matrix, observations=None, parameters=None, ard=True, hyper_shape_0=None,
                 hyper_scale_0=None, weights=None):
        # Copy all the data
        self.design_matrix = np.atleast_2d(design_matrix)
        self.n, self.p = self.design_matrix.shape
        self.parameters = parameters if parameters is None else np.atleast_1d(parameters)
        self.weights = np.ones(self.n) if weights is None else np.atleast_1d(weights)
        self.ard = ard
        self.hyper_shape_0 = hyper_shape_0 or 1e-2
        self.hyper_scale_0 = hyper_scale_0 or 1e-4
        if observations is not None:
            self.observations = np.asarray(observations)
            # Recode as +-1 if necessary
            if self.observations.dtype == np.bool:
                self.observations = 2 * self.observations - 1
            assert np.all(np.abs(self.observations) == 1), \
                "Observations must be coded as +1 for success and -1 for failure."
        else:
            self.observations = None

    def predictor(self, parameters, design_matrix):
        """
        Evaluate the predictor before application of the sigmoid.
        """
        return np.dot(design_matrix, parameters)

    def likelihood(self, parameters=None, design_matrix=None, observations=None):
        """
        Evaluate the likelihood.
        """
        raise NotImplementedError

    def log_likelihood(self, parameters=None, design_matrix=None, observations=None):
        """
        Evaluate the log likelihood.
        """
        return np.log(self.likelihood(parameters, design_matrix, observations))

    def forward(self):
        """
        Generate observations.
        """
        probability = self.likelihood(self.parameters, self.design_matrix, 1)
        # Sample observations and encode them as [-1, 1]
        self.observations = 2 * (np.random.uniform(size=self.n) < probability) - 1
        return self.observations

    def variational_step(self, **kwargs):
        raise NotImplementedError

    def variational(self, tol=1e-5, maxiter=1000, report=None, log=1, log_keys=None, **kwargs):
        # Add zero parameter means if they are not given
        kwargs['parameter_means'] = np.asarray(kwargs.get('parameter_means', np.zeros(self.p)))

        # Initialise parameters for iteration
        elbo = None
        previous_elbo = -np.inf
        converged=False
        step = 0
        elbos = []
        log_keys = ['parameter_means', 'parameter_cov'] if log_keys is None else log_keys
        results = {key: [] for key in log_keys}

        while not converged:
            if step >= maxiter:
                warn("Number of steps exceeded `maxiter={}`.".format(maxiter))
                break

            if report is not None and (step + 1) % report == 0:
                print step + 1, elbo, kwargs

            elbo, kwargs = self.variational_step(**kwargs)
            if log > 0 and step % log == 0:
                elbos.append(elbo)
                for key in log_keys:
                    results[key].append(kwargs[key])
            step += 1

            # Check whether the desired tolerance has been achieved
            if elbo - previous_elbo < tol * np.abs(previous_elbo):
                break
            previous_elbo = elbo

        return (elbos, {k: np.asarray(v) for k,v in results.iteritems()}) if log > 0 \
            else (elbo, {key: kwargs[key] for key in log_keys})

    def gibbs_step(self, **kwargs):
        raise NotImplementedError

    def gibbs(self, steps, report=None, log=1, log_keys=None, **kwargs):
        """
        Perform inference using a Gibbs sampler.
        """
        # Add zero parameter means if they are not given
        kwargs['parameters'] = np.asarray(kwargs.get('parameters', np.zeros(self.p)))

        log_keys = ['parameters'] if log_keys is None else log_keys
        results = {key: [] for key in log_keys}

        for step in range(steps):
            kwargs = self.gibbs_step(**kwargs)

            if report is not None and (step + 1) % report == 0:
                print step + 1, kwargs

            if log > 0 and step % log == 0:
                for key in log_keys:
                    results[key].append(kwargs[key])

        return {k: np.asarray(v) for k, v in results.iteritems()}


class LogisticRegression(BinaryRegression):
    def likelihood(self, parameters=None, design_matrix=None, observations=None):
        """
        Evaluate the likelihood.
        """
        # Get default values if available
        parameters = parameters if parameters is not None else self.parameters
        design_matrix = design_matrix if design_matrix is not None else self.design_matrix
        observations = observations if observations is not None else self.observations
        return logistic(observations * self.predictor(parameters, design_matrix))

    def log_likelihood(self, parameters=None, design_matrix=None, observations=None):
        """
        Evaluate the log likelihood.
        """
        # Get default values if available
        parameters = parameters if parameters is not None else self.parameters
        design_matrix = design_matrix if design_matrix is not None else self.design_matrix
        observations = observations if observations is not None else self.observations
        return -np.logaddexp(0, -observations * design_matrix.dot(parameters))

    def variational_step(self, **kwargs):
        """
        Infer parameters from observations.
        """
        parameter_means = kwargs['parameter_means']
        # Get lambda_xi if not given
        if 'xi' in kwargs:
            lambda_xi = logistic_lambda(kwargs['xi'])
        elif 'lambda_xi' in kwargs:
            lambda_xi = kwargs['lambda_xi']
        else:
            lambda_xi = 0.125

        # Compute the expected prior precision
        if self.ard:
            hyper_shape = self.hyper_shape_0 + 0.5
            hyper_scale = self.hyper_scale_0 + 0.5 * parameter_means * parameter_means
        else:
            hyper_shape = self.hyper_shape_0 + 0.5 * self.p
            hyper_scale = self.hyper_scale_0 + 0.5 * parameter_means.dot(parameter_means)
        tau = hyper_shape / hyper_scale

        # Compute the parameter covariance and mean
        parameter_precision = tau * np.eye(self.p) + 2 * np.dot(self.design_matrix.T * lambda_xi * self.weights,
                                                                self.design_matrix)
        parameter_cov = np.linalg.inv(parameter_precision)
        parameter_means = 0.5 * parameter_cov.dot((self.observations * self.weights).dot(self.design_matrix))

        # Evaluate the extra variational parameter
        xi = np.sqrt(np.sum(np.dot(self.design_matrix, parameter_cov + parameter_means[:, None] *
                                   parameter_means[None, :]) * self.design_matrix, axis=1))
        logistic_xi = logistic(xi)
        lambda_xi = logistic_lambda(xi, logistic_xi)

        # Evaluate the evidence lower bound
        elbo = .5 * parameter_means.dot(parameter_precision).dot(parameter_means) \
                    - .5 * np.log(np.linalg.det(parameter_precision)) \
                    + np.sum(np.log(logistic_xi) - .5 * xi + lambda_xi * xi ** 2) \
                    + np.sum(- gammaln(self.hyper_shape_0) + self.hyper_shape_0 * np.log(self.hyper_scale_0)
                             - self.hyper_scale_0 * hyper_shape / hyper_scale - hyper_shape * np.log(hyper_scale)
                             + gammaln(hyper_shape) + hyper_shape)

        return elbo, dict(parameter_means=parameter_means, xi=xi, lambda_xi=lambda_xi, parameter_cov=parameter_cov)


class ProbitRegression(BinaryRegression):
    def likelihood(self, parameters=None, design_matrix=None, observations=None):
        """
        Evaluate the likelihood.
        """
        # Get default values if available
        parameters = parameters if parameters is not None else self.parameters
        design_matrix = design_matrix if design_matrix is not None else self.design_matrix
        observations = observations if observations is not None else self.observations
        return normal_cdf(observations * self.predictor(parameters, design_matrix))

    def gibbs_step(self, **kwargs):
        parameters = kwargs['parameters']
        # Update the hyperparameters
        if self.ard:
            hyper_shape = self.hyper_shape_0 + 0.5
            hyper_scale = self.hyper_scale_0 + 0.5 * parameters * parameters
        else:
            hyper_shape = self.hyper_shape_0 + 0.5 * self.p
            hyper_scale = self.hyper_scale_0 + 0.5 * parameters.dot(parameters)
        # Sample the prior precision
        tau = np.random.gamma(hyper_shape, 1.0 / hyper_scale)

        # Evaluate the untruncated means of the latent variables z
        mean_z = self.design_matrix.dot(parameters)
        # Draw samples of the latent variables z
        z = self.observations * truncnorm(-self.observations * mean_z) + mean_z

        # Compute the precision and mean for the parameters
        parameter_precision = tau * np.eye(self.p) + (self.design_matrix.T * self.weights).dot(self.design_matrix)
        parameter_cov = np.linalg.inv(parameter_precision)
        parameter_mean = parameter_cov.dot(self.design_matrix.T * self.weights).dot(z)

        # Sample the parameters
        parameters = np.random.multivariate_normal(parameter_mean, parameter_cov)
        return dict(parameters=parameters)

    def variational_step(self, **kwargs):
        parameter_means = kwargs['parameter_means']
        # Compute the expected prior precision
        if self.ard:
            hyper_shape = self.hyper_shape_0 + 0.5
            hyper_scale = self.hyper_scale_0 + 0.5 * parameter_means * parameter_means
        else:
            hyper_shape = self.hyper_shape_0 + 0.5 * self.p
            hyper_scale = self.hyper_scale_0 + 0.5 * parameter_means.dot(parameter_means)
        tau = hyper_shape / hyper_scale

        # Compute the untruncated means
        z = self.design_matrix.dot(parameter_means)
        # Compute the truncated means
        z += self.observations * inverse_mills_ratio(self.observations * z)

        # Compute the precision matrix and mean
        parameter_precision = np.eye(self.p) * tau + (self.design_matrix.T * self.weights).dot(self.design_matrix)
        parameter_cov = np.linalg.inv(parameter_precision)
        parameter_means = parameter_cov.dot(self.design_matrix.T * self.weights).dot(z)

        # Compute the evidence lower bound
        elbo = 0

        return elbo, dict(parameter_means=parameter_means, parameter_cov=parameter_cov)


def __main__():
    import matplotlib.pyplot as plt
    from scipy import stats

    n = 1000
    p = 8
    s = 1
    mode = 'probit'

    np.random.seed(s)
    seed_rng(s)
    design_matrix = np.random.normal(0, 1, (n, p))
    parameters = (-1, 2)
    parameters = np.concatenate((parameters, np.zeros(p - len(parameters))))
    # Generate some data
    if mode=='probit':
        regression = ProbitRegression(design_matrix, parameters=parameters, ard=True)
    elif mode=='logistic':
        regression = LogisticRegression(design_matrix, parameters=parameters, ard=True)
    else:
        raise ValueError
    regression.forward()
    # Infer the results
    samples = regression.gibbs(2000) if mode=='probit' else None

    # Same with variational results
    elbos, sequence = regression.variational()

    # Show trace plots
    plt.subplot(121)
    colors = 'rbgk'
    j = 0
    for i, parameter in enumerate(parameters):
        if parameter:
            color = colors[j % len(colors)]
            j+=1
        else:
            color='gray'
        plt.axhline(parameter, color=color, ls='dashed')
        if samples is not None:
            plt.plot(samples['parameters'][:,i], color=color, ls=':')
        plt.plot(sequence['parameter_means'][:,i], color=color)

    plt.subplot(122)
    plt.plot(elbos)
    plt.show()

    # Show uncertainties
    rows, cols = 2, 4
    j=0
    for i in range(p):
        if parameters[i]:
            color = colors[j % len(colors)]
            j+=1
        else:
            color='gray'

        ax = plt.subplot(rows, cols, i + 1)
        if samples is not None:
            # Show a histogram
            ax.hist(samples['parameters'][500:, i], normed=True, histtype='stepfilled', color=color, alpha=.2)
        # Show the variational approximation
        mean, std = sequence['parameter_means'][-1, i], np.sqrt(sequence['parameter_cov'][-1, i, i])
        linx = np.linspace(mean - 3 * std, mean + 3 * std)
        ax.plot(linx, stats.norm.pdf(linx, mean, std), color=color)
        ax.axvline(parameters[i], color=color)
        ax.axvline(0, color='k', ls='dotted')

    plt.show()


if __name__ == '__main__':
    __main__()