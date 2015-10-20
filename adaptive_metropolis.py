import numpy as np


class AdaptiveMetropolisSampler:
    def __init__(self, parameters, log_posterior, threshold=100,
                 epsilon=1e-5, scale=5.76, mode='update'):
        # Store information
        self.parameters = np.atleast_1d(parameters)
        self.log_posterior = log_posterior
        self.threshold = threshold
        self.epsilon = epsilon
        self.mode = mode

        # Store derived information
        self.num_parameters = self.parameters.shape[0]
        # Initialise the proposal scale
        self.scale = scale / self.num_parameters
        # Initialise the covariance matrix
        self.covariance0 = epsilon * np.eye(self.num_parameters)
        # Initialise the running mean and variance
        self.sample_mean = np.zeros(self.num_parameters)
        self.sample_covariance = 0
        # Initialise containers for the samples and log posterior
        self.samples = []
        self.log_posteriors = []

    def acceptance_rate(self, burnin=0):
        """
        Computes the acceptance rate.
        """
        samples = np.asarray(self.samples[burnin:])
        return np.mean(samples[1:] != samples[:-1])

    def __call__(self, num_steps, report=0, **kwargs):
        """
        Performs the specified number of Metropolis-Hastings steps.
        :param num_steps: The number of steps to make.
        :return: All samples of the parameter values including samples from previous calls.
        """
        # Initialise log posterior
        if self.mode=='update':
            lp_current = self.log_posterior(self.parameters, **kwargs)

        # Iterate over steps
        for step in range(num_steps):
            # Make a proposal with the initial covariance or the scaled sample covariance
            _covariance = self.covariance0 if len(self.samples) < self.threshold \
                else self.sample_covariance + self.covariance0
            proposal = np.random.multivariate_normal(self.parameters, self.scale * _covariance)
            # Compute the log posterior
            if self.mode=='update':
                lp_proposal = self.log_posterior(proposal, **kwargs)
            elif self.mode=='reevaluate':
                lp_current, lp_proposal = self.log_posterior(proposal, self.parameters, **kwargs)
            else:
                raise KeyError(self.mode)
            # Accept or reject the step
            if lp_proposal - lp_current > np.log(np.random.uniform()):
                # Update the log posterior and the parameter values
                lp_current = lp_proposal
                self.parameters = proposal

            # Update the sample mean...
            previous_mean = self.sample_mean
            self.sample_mean = (self.parameters + len(self.samples) * previous_mean) / (len(self.samples) + 1)

            # ...and the sample covariance
            self.sample_covariance = (len(self.samples) * self.sample_covariance + self.parameters *
                                      self.parameters[:, None] + len(self.samples) * previous_mean *
                                      previous_mean[:, None]) / (len(self.samples) + 1) - self.sample_mean * \
                                                                                          self.sample_mean[:, None]

            # Save the parameters
            self.samples.append(self.parameters)
            self.log_posteriors.append(lp_current)
    
            # Report if desired
            if report and (step + 1) % report==0:
                print step + 1, self.acceptance_rate(step + 1 - report)

        return np.asarray(self.samples)


def __main__():
    from matplotlib import pyplot as plt
    # Generate test data from a multivariate Gaussian
    num_samples = 100
    mu = [3, 4]
    sigma = np.eye(len(mu))
    data = np.random.multivariate_normal(mu, sigma, num_samples)

    # Define the log posterior
    def log_posterior(_parameters, **kwargs):
        residuals = kwargs['data'] - _parameters
        return -5 * np.sum(residuals * residuals)

    # Initialise the adaptive metropolis sampler
    ams = AdaptiveMetropolisSampler(mu, log_posterior)
    # Obtain 2000 samples
    samples = ams(2000, data=data)

    # Create a trace plot
    fig = plt.figure()
    colours = "rgb"
    ax1 = fig.add_subplot(111)
    for i in range(len(mu)):
        ax1.plot(samples[:, i], color=colours[i])
        ax1.axhline(mu[i], color=colours[i])
        ax1.axhline(np.mean(data[:, i]), color=colours[i], ls='--')

    plt.show()
    print "Acceptance rate: {}".format(ams.acceptance_rate())


if __name__ == '__main__':
    __main__()