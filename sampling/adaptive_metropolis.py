import numpy as np
from base import BaseSampler, normal_log_posterior


class AdaptiveMetropolisSampler(BaseSampler):
    def __init__(self, fun, args=None, parameter_names=None, threshold=100, epsilon=1e-5,
                 scale=5.76, mode='update'):
        super(AdaptiveMetropolisSampler, self).__init__(fun, args, parameter_names)

        self.threshold = threshold
        self.epsilon = epsilon
        self.mode = mode

        self.num_parameters = None
        self.scale = scale

    def sample(self, parameters, steps=1, callback=None):
        parameters = np.asarray(parameters)

        if self.num_parameters is None:
            # Store derived information
            self.num_parameters = len(parameters)
            # Initialise the proposal scale
            self.scale = self.scale / self.num_parameters
            # Initialise the covariance matrix
            self.covariance0 = self.epsilon * np.eye(self.num_parameters)
            # Initialise the running mean and variance
            self.sample_mean = np.zeros(self.num_parameters)
            self.sample_covariance = 0

        for step in range(steps):
            if len(self._fun_values) == 0 or self.mode=='reevaluate':
                lp_current = self.fun(parameters, *self.args)
            else:
                lp_current = self._fun_values[-1]

            # Make a proposal with the initial covariance or the scaled sample covariance
            _covariance = self.covariance0 if len(self._samples) < self.threshold \
                else self.sample_covariance + self.covariance0
            proposal = np.random.multivariate_normal(parameters, self.scale * _covariance)
            # Compute the log posterior
            lp_proposal = self.fun(proposal, *self.args)
            # Accept or reject the step
            if lp_proposal - lp_current > np.log(np.random.uniform()):
                # Update the log posterior and the parameter values
                lp_current = lp_proposal
                parameters = proposal

            # Update the sample mean...
            previous_mean = self.sample_mean
            self.sample_mean = (parameters + len(self._samples) * previous_mean) / (len(self._samples) + 1)

            # ...and the sample covariance
            self.sample_covariance = (len(self._samples) * self.sample_covariance + parameters *
                                      parameters[:, None] + len(self._samples) * previous_mean *
                                      previous_mean[:, None]) / (len(self._samples) + 1) - self.sample_mean * \
                                                                                          self.sample_mean[:, None]

            # Save the parameters
            self._samples.append(parameters)
            self._fun_values.append(lp_current)

            if callable(callback):
                callback(parameters)

        return parameters


def __main__():
    from matplotlib import pyplot as plt
    np.random.seed(1)

    mean = np.asarray([-1, 1, 3])
    parameter_names = [r'$\mu_{{{0}}}$'.format(i + 1) for i in range(len(mean))]

    # Initialise the adaptive metropolis sampler
    sampler = AdaptiveMetropolisSampler(normal_log_posterior, (mean,), parameter_names)
    # Obtain 2000 samples
    sampler.sample(mean, 2000)

    burn_in = 500
    sampler.describe(burn_in)
    sampler.trace_plot(burn_in)
    sampler.density_plot(burn_in)

    plt.show()

    return sampler


if __name__ == '__main__':
    sampler = __main__()