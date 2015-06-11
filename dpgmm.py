import numpy as np
from scipy.stats import wishart


def weighted_mean(X, weight):
    """
    Evaluate the weighted mean of the data.
    
    Parameters
    ----------
    X : array_like, 2 dimensions
        A `n` by `ndim` array of `n` observations in `ndim` dimensions.
    weight : array_like, 1 dimension
        A length `n` array of observation weights.
    """
    return np.sum(X * weight[:, None], axis=0) / np.sum(weight)


def weighted_cov(X, weight, mu=None):
    """
    Evaluate the weighted, biased covariance of the data.
    
    Parameters
    ----------
    X : array_like, 2 dimensions
        A `n` by `ndim` array of `n` observations in `ndim` dimensions.
    weight : array_like, 1 dimension
        A length `n` array of observation weights.
    """
    # Evaluate the mean if necessary
    if mu is None:
        mu = weighted_mean(X, weight)
    # Compute residuals
    X = X - mu
    return np.atleast_2d(np.dot(X.T * weight, X)) / np.sum(weight)


class Component:
    """
    Single component of a Gaussian mixture model.

    Parameters
    ----------
    mixture : `Mixture` instance
        The mixture to which the component belongs.
    idx : `list` of integers
        A list of data point indices that belong to the component.

    Attributes
    ----------
    idx : `list` of integers
        A list of data point indices that belong to the component.
    mixture : `Mixture` instance
        The mixture to which the component belongs.
    total_weight : float
        Total weight of associated data points.
    pkappa : float
        Precision parameter for NIW posterior on component mean.
    pnu : float
        Degrees of freedoms for NIW posterior on component precision.
    marginal_det : float
        Determinant of the marginal covariance matrix.
    pchi : array, 1 dimension
        Location parameter for NIW posterior on component mean.
    data_mean : array, 1 dimension
        Mean of associated data points.
    data_cov : array, 2 dimensions
        Covariance of associated data points.
    ppsi : array, 2 dimensions
        Scale parameter for NIW posterior on component precision.
    inv_marginal_cov : array, 2 dimensions
        Inverse of marginal covariance matrix.
    """
    def __init__(self, mixture, idx):
        # Store the list of indices of data points in this component
        self.idx = idx
        assert type(self.idx) == list, "`idx` must be of type `list`."
        # Store the mixture to which this component belongs
        self.mixture = mixture
        # Declare instance attributes
        self.total_weight = self.pkappa = self.pnu = self.marginal_det = 0
        self.data_mean = self.pchi = np.zeros(self.mixture.ndim)
        self.data_cov = self.ppsi = self.inv_marginal_cov = np.zeros((self.mixture.ndim, self.mixture.ndim))
        # Compute initial statistics
        self.update_statistics()

    def remove(self, idx):
        """
        Remove the specified data point from the component.

        Parameters
        ----------
        idx : int
            Index of the data point to remove.
        """
        self.idx.remove(idx)
        self.update_statistics()

    def append(self, idx):
        """
        Add the specified data point from the component.

        Parameters
        ----------
        idx : int
            Index of the data point to add.
        """
        self.idx.append(idx)
        self.update_statistics()

    def update_statistics(self):
        """
        Updates the summary statistics of the component including
        `data_mean`, `data_cov`, `total_weight`, `pnu`, `pchi`, 
        `pkappa`, `ppsi`, `marginal_det`, `inv_marginal_cov`.
        """
        if len(self.idx) == 1:
            # Just store the value of the single point
            self.data_mean = self.mixture.X[self.idx[0]]
            # One data point has zero variance
            self.data_cov = np.zeros((self.mixture.ndim, self.mixture.ndim))
            # Store the weight of the single point
            self.total_weight = self.mixture.weight[self.idx[0]]
        elif len(self.idx) > 1:
            # Extract the cluster data and weight
            X = self.mixture.X[self.idx]
            weight = self.mixture.weight[self.idx]
            self.total_weight = np.sum(weight)
            # Compute the mean
            self.data_mean = weighted_mean(X, weight)
            # Compute the (biased) covariance
            self.data_cov = weighted_cov(X, weight)

        if len(self.idx) == 0:
            # Use very diffuse priors if there is no data in the component
            self.pchi = self.mixture.data_mean
            marginal_cov = 1e6 * self.mixture.data_cov
        else:
            # Compute the posterior parameters (cf. Sec. 8.3
            # http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
            self.pkappa = self.mixture.kappa + self.total_weight
            self.pnu = self.mixture.nu + self.total_weight
            self.pchi = (self.mixture.kappa * self.mixture.chi
                         + self.total_weight * self.data_mean) / self.pkappa
            self.ppsi = self.mixture.psi + self.total_weight * self.data_cov \
                + self.mixture.kappa * self.total_weight / self.pkappa \
                * np.sum((self.mixture.chi - self.data_mean) ** 2)

            # Compute the marginal covariance
            marginal_cov = (self.ppsi * (self.pkappa + 1)) \
                / (self.pkappa * (self.pnu - self.mixture.ndim + 1))
            # The marginal mean is simply `pchi` so no need to calculate

        # Obtain the inverse and determinant
        self.inv_marginal_cov = np.linalg.inv(marginal_cov)
        self.marginal_det = np.linalg.det(marginal_cov)

    def marginal_likelihood(self, x):
        """
        Evaluates the unnormalised marginal likelihood.
        
        Parameters
        ----------
        x : float
            Point at which to evaluate.
        """
        assert len(x) == self.mixture.ndim, "`x` must have the same dimension as the data."
        # Compute the residuals
        residual = x - self.pchi
        # Return the result
        return np.exp(-0.5 * residual.dot(self.inv_marginal_cov).dot(residual)) / np.sqrt(self.marginal_det)

    
class Mixture:
    """
    Dirichlet process Gaussian mixture with a variable number of components.
    
    Implementation of a Gibbs sampling algorithm to perform nonparametric
    Bayesian density estimation according to algorithm 3 of [2]. The code 
    is based on [3] but has significant performance improvements and allows
    the evaluation of the mixture at every sampling step.
    
    Example
    -------
    import numpy as np
    
    # Generate data from two Gaussian components
    X = np.concatenate((np.random.normal(-2, 1, 100),
                        np.random.normal(2, 1, 50)))
    # Create a mixture model and run some steps
    mixture = Mixture(X)
    mixture.make_steps(50)
    # Print the number of components as sampling progresses
    print [len(rho) for rho in mixture.samples_rho]
    
    Notes
    -----
    The default value for the prior parameter `psi` is technically a cheat
    because it depends on the covariance matrix of the data. Ideally, we
    would like `psi` to be small because the posterior is otherwise dominated
    by the prior parameter. However, convergence is extremely slow if `psi`
    is too small. Letting the prior depend on the data is a compromise.

    Parameters
    ----------
    X : array_like, 2 dimensions
        A `n` by `ndim` array of `n` observations in `ndim` dimensions.
    alpha : float
        Dirichlet process concentration parameter [1].
    chi : array, 1 dimension
        Location parameter for NIW prior on component mean.
    kappa : float
        Precision parameter for NIW prior on component mean.
    psi : array, 2 dimensions
        Scale parameter for NIW prior on component precision.
    nu : float
        Degrees of freedoms for NIW prior on component precision.
    weight : array_like, 1 dimension
        A length `n` array of observation weights.

    Attributes
    ----------
    X : array_like, 2 dimensions
        A `n` by `ndim` array of `n` observations in `ndim` dimensions.
    alpha : float
        Dirichlet process concentration parameter [1]. 
        Default: 1.0
    chi : array, 1 dimension
        Location parameter for NIW prior on component mean. 
        Default: mean of the data
    kappa : float
        Precision parameter for NIW prior on component mean.
        Default: 0.0
    psi : array, 2 dimensions
        Scale parameter for NIW prior on component precision.
        Default: (data covariance) / (number of observations)
    nu : float
        Degrees of freedoms for NIW prior on component precision.
        Default: number of dimensions
    weight : array_like, 1 dimension
        A length `n` array of observation weights.
        Default: unit vector
    npoints : int
        Number of data points.
    ndim : int
        Number of dimensions of observations
    components : dict of Component instances
        Dictionary of instances of components keyed by component
        identifier.
    z : dict of integers
        Dictionary of component identifiers keyed by data point
        index.
    samples_rho : list
        Samples of component weights.
    samples_mu : list
        Samples of component means.
    samples_tau : list
        Samples of component precision matrices.
    data_mean : array, 1 dimension
        Mean of the data.
    data_cov : array, 2 dimensions
        Covariance of the data.
        
    References
    ----------
    [1] C. E. Rasmussen. The infinite gaussian mixture model. In S. Solla, 
        T. Leen, and K.-R. M\"uller, editors, Advances in Neural Information 
        Processing Systems 12, pages 554--560. MIT Press, 2000.
    [2] R. M. Neal. Markov chain sampling methods for dirichlet process
        mixture models. Journal of Computational and Graphical
        Statistics, 9(2):249--265, 2000.     
    [3] G. Synnaeve. Collapsed Gibbs Sampling for Dirichlet Process
        Gaussian Mixture Models. URL: http://bit.ly/1FaL8zE
    """
    def __init__(self, X, alpha=1.0, chi=None, kappa=0.0, psi=None, nu=None, weight=None):
        self.alpha = alpha
        # Store the data array
        self.X = np.asarray(X)
        if self.X.ndim == 1:
            self.X = self.X[:, None]
        assert self.X.ndim == 2, "`X` must have two dimensions."
        self.npoints, self.ndim = self.X.shape
        # Save the weights
        self.weight = np.ones(self.npoints) if weight is None else np.asarray(weight)
        # Update summary stats
        self.data_mean = weighted_mean(self.X, self.weight)
        self.data_cov = weighted_cov(self.X, self.weight, self.data_mean)
        # Save the prior mean expectation
        self.chi = np.mean(self.X, axis=0) if chi is None else chi
        assert self.chi.shape[0] == self.X.shape[1], \
            "`chi` must have the same length as the number of dimensions."
        # Save the prior mean precision
        self.kappa = kappa
        # Save the prior covariance expectation (bias=1 for proper normalisation)
        self.psi = (self.data_cov / self.npoints) if psi is None else psi
        # Store degrees of freedom
        self.nu = max(nu, self.ndim)
        # Assign to components
        self.components = {i: Component(self, [i]) for i in range(self.npoints)}
        # Keep a dictionary of components
        self.z = {i: i for i in range(self.npoints)}

        # Create sample containers (mixture, size, means, precision matrices)
        self.samples_rho = []
        self.samples_mu = []
        self.samples_tau = []

    def make_steps(self, num_steps):
        """
        Make multiple Gibbs sampling steps:
        
        Parameters
        ----------
        num_steps : int
            Number of steps to make.
        """
        for step in range(num_steps):
            self.make_step()

    def make_step(self):
        """
        Make a single Gibbs sampling step according to [2] algorithm 3.
        """
        # Create an empty base component to evaluate the probability of new components
        base_component = Component(self, [])
        # Iterate over all observations
        for i in range(self.npoints):
            # Remove the observation from its current cluster
            z_i = self.z[i]
            self.components[z_i].remove(i)

            # Remove the component if it is empty
            if len(self.components[z_i].idx) == 0:
                del self.components[z_i]

            # Iterate over all possible components and evaluate the
            # probability to assign the current data point to the cluster
            probabilities = []
            labels = []
            for z, component in self.components.iteritems():
                # Obtain the marginal likelihood under the Gaussian
                p = component.marginal_likelihood(self.X[i])
                # Obtain the contribution from the Dirichlet process
                p *= component.total_weight / (self.alpha + self.npoints - 1.0)
                # Append to the list of probabilities and labels
                probabilities.append(p)
                labels.append(z)

            # Consider the possibility of a new component
            # Define a new label
            z_new = np.max(self.components.keys()) + 1
            # Probabilities for adding a new cluster
            p = base_component.marginal_likelihood(self.X[i])
            p *= self.alpha / (self.alpha + self.npoints - 1.0)
            # Append to the list of probabilities and labels
            probabilities.append(p)
            labels.append(z_new)

            # Normalise the distribution
            probabilities = np.asarray(probabilities) / np.sum(probabilities)
            # Sample a new cluster
            self.z[i] = z_i = np.random.choice(labels, p=probabilities)
            # If it's a new cluster
            if z_i == z_new:
                self.components[z_i] = Component(self, [i])
            else:
                self.components[z_i].append(i)

        # Sample the density of each cluster
        self.samples_rho.append(np.random.dirichlet(self.alpha
                                + np.asarray([component.total_weight for component in self.components.itervalues()])))

        # Sample the inverse covariance
        row_mu = []
        row_tau = []
        for component in self.components.itervalues():
            # Draw from the Wishart distribution
            tau = np.atleast_2d(wishart.rvs(df=component.pnu, scale=np.linalg.inv(component.ppsi)))
            # Draw from the multivariate normal
            mu = np.random.multivariate_normal(component.pchi, np.linalg.inv(component.pkappa * tau))
            # Append samples
            row_mu.append(mu)
            row_tau.append(tau)
        # Store the results
        self.samples_mu.append(row_mu)
        self.samples_tau.append(row_tau)

    def evaluate(self, Y):
        """
        Evaluate samples of the mixture distribution.
        
        Parameters
        ----------
        Y : array_like, 2 dimensions
            A `m` by `ndim` array of `m` points in `ndim` dimensions at which to evaluate the mixture.
            
        Returns
        -------
        ndarray, 2 dimensions
            A `ns` by `m` array of `ns` samples of the mixture distribution.
        """
        # Bring data to the right shape
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y[:, None]

        # Iterate over all samples
        samples = []
        for list_rho, list_mu, list_tau in zip(self.samples_rho, self.samples_mu,
                                               self.samples_tau):
            # Iterate over all components
            sample = 0
            for rho, mu, tau in zip(list_rho, list_mu, list_tau):
                det = np.linalg.det(tau)
                norm = (2 * np.pi) ** (0.5 * self.ndim) / np.sqrt(det)
                # This slicing should ensure that we have a m by d object
                # where m is the number of evaluation points and d is the
                # dimensionality
                residual = mu - Y
                # The chi2 for an individual entry y = Y[i] is (in Einstein notation)
                # chi2 = y[j]tau[j,k]y[k]
                # For a list of chi2 we have
                # chi2[i] = Y[i, j]tau[j,k]Y[i,k]
                #         = Y[i, j]tau[j,k]Y.T[k, i]
                chi2 = np.sum(residual.dot(tau) * residual, axis=1)
                # Add the weighted contribution
                sample += rho * np.exp(-0.5 * chi2) / norm
            samples.append(sample)
        return np.asarray(samples)