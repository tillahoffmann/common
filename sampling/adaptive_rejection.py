import numpy as np
import bisect
from matplotlib import pyplot as plt


class AdaptiveRejectionSampler:
    """
    Adaptive rejection sampler to draw samples from log-concave distributions.

    Parameters
    ----------
    fun : callable
        function to compute the (unnormalised) log-PDF and its derivative if `jac` is not given
    x0 : float or sequence of floats, optional
        abscissa or abscissas to initialise the sampler with
    args : tuple, optional
        extra arguments passed to the log-likelihood function
    jac : callable, optional
        function to compute the derivative of the log-PDF. If `jac` is not given, `fun` must return the function value
        and the derivative.
    domain : (float, float), optional
        lower and upper bound of the domain on which the distribution is supported. If `domain` is not given, the
        domain is the positive real line.
    """

    def __init__(self, fun, x0=None, args=None, jac=None, domain=None):
        # Copy the function and the Jacobian
        self.fun = fun
        self.jac = jac
        self.domain = (-np.inf, np.inf) if domain is None else domain
        # Store the extra arguments for the function
        self.args = tuple() if args is None else args
        # Create containers for the abscissas, function values and derivatives
        self.abscissas = []
        self.fun_values = []
        self.jac_values = []
        # Insert the domain values into the intersection points
        self.hull_abscissas = []
        # The weight of each segment
        self.hull_weights = []
        # Define a maximum value as an offset (we don't care about the overall scale and want to avoid numeric problems)
        self._fun_maximum = -np.inf
        # Add the initial value(s)
        if hasattr(x0, '__iter__'):
            self.add_abscissas(x0)
        elif x0 is not None:
            self.add_abscissa(x0)

    def _hull_intersection(self, idx):
        """
        Evaluate the intersection point between two adjacent tangents.

        Parameters
        ----------
        idx : int
            index of the first (smaller) abscissa

        Returns
        -------
        intersection : float
            location at which the tangents intersect
        """
        # Compute the intersection according to Eq. (1) in Gilks and Wild
        intersection = (self.fun_values[idx + 1] - self.fun_values[idx] - self.abscissas[idx + 1] *
                        self.jac_values[idx + 1] + self.abscissas[idx] * self.jac_values[idx]) / \
                       (self.jac_values[idx] - self.jac_values[idx + 1])
        assert self.abscissas[idx] < intersection < self.abscissas[idx + 1], "intersection must lie between abscissas"
        return intersection

    def _hull_weight(self, idx):
        """
        Evaluate the (unnormalised) probability weight of a tangent section.

        Parameters
        ----------
        idx : int
            index of the tangent segment

        Returns
        -------
        weight : float
            unnormalised probability weight of the tangent

        Notes
        -----
        The upper hull of the `idx` segment is a linear function as defined in Eq. (2) in Gilks and Wild. The integral
        can be obtained analytically.
        """
        # Compute the integral of the tangent associated with the point at `idx`
        fun_value = self.fun_values[idx]
        jac_value = self.jac_values[idx]
        lower = self.hull_abscissas[idx]
        upper = self.hull_abscissas[idx + 1]
        abscissa = self.abscissas[idx]
        weight = np.exp(fun_value - jac_value * abscissa) * (np.exp(upper * jac_value) - np.exp(lower * jac_value)) / \
            jac_value

        return weight

    def add_abscissa(self, abscissa):
        """
        Add an abscissa.

        Parameters
        ----------
        abscissa : float
            location of the abscissa

        Returns
        -------
        fun_value : float
            value of the log-PDF up to an additive constant
        jac_value : float
            derivative of the log-PDF
        """
        # Check that the abscissa is not outside the domain
        if abscissa < self.domain[0] or abscissa > self.domain[1]:
            raise ValueError("abscissa {0} falls outside the domain [{1}, {2}]".format(abscissa, *self.domain))

        # Identify where the abscissa should be placed
        idx = bisect.bisect_left(self.abscissas, abscissa)
        # Insert the abscissa
        self.abscissas.insert(idx, abscissa)

        # Evaluate the function and its derivative
        if self.jac:
            fun_value = self.fun(abscissa, *self.args)
            jac_value = self.jac(abscissa, *self.args)
        else:
            fun_value, jac_value = self.fun(abscissa, *self.args)

        # Check if the new value exceeds the previous maximum
        if fun_value > self._fun_maximum:
            # Compute the difference
            delta = fun_value - self._fun_maximum
            self._fun_maximum = fun_value
            # Update all the other values
            self.fun_values = map(lambda x: x - delta, self.fun_values)
            # Rescale all the weights
            self.hull_weights = map(lambda x: x / np.exp(delta), self.hull_weights)

        # Add the function values after additive normalisation
        fun_value -= self._fun_maximum
        self.fun_values.insert(idx, fun_value)
        self.jac_values.insert(idx, jac_value)

        # Delete the hull abscissa to the left if it exists
        if self.hull_abscissas:
            del self.hull_abscissas[idx]

        # Add the abscissa to the right (use the upper domain limit if it's the last abscissa)
        if idx == len(self.abscissas) - 1:
            self.hull_abscissas.insert(idx, self.domain[1])
        else:
            # Compute the intersection with the next point if it is an interior point
            self.hull_abscissas.insert(idx, self._hull_intersection(idx))

        # Add the abscissa to the left (use the lower domain limit if it's the first abscissa)
        if idx == 0:
            self.hull_abscissas.insert(idx, self.domain[0])
        else:
            # Compute the intersection with the previous point if it is an interior point
            self.hull_abscissas.insert(idx, self._hull_intersection(idx - 1))

        # Evaluate the hull weight
        self.hull_weights.insert(idx, self._hull_weight(idx))

        # Update the hull weight of the previous segment ...
        if idx > 0:
            self.hull_weights[idx - 1] = self._hull_weight(idx - 1)

        # ... and the next segment
        if idx < len(self.abscissas) - 1:
            self.hull_weights[idx + 1] = self._hull_weight(idx + 1)

        return fun_value, jac_value

    def add_abscissas(self, abscissas):
        """
        Add multiple abscissas.

        Parameters
        ----------
        abscissas : array_like
            locations of abscissas

        Returns
        -------
        fun_values : array_like
            values of the log-PDF up to an additive constant
        jac_values : array_like
            derivatives of the log-PDF
        """
        return np.transpose(map(self.add_abscissa, abscissas))

    def plot(self, start=None, stop=None, num=50, ax=None):
        """
        Plot the state of the sampler.

        Parameters
        ----------
        start : float, optional
            start of the value sequence to plot
        stop : float, optional
            end of the value sequence to plot
        num : int, optional
            number of sample points
        ax : plt.Axes, optional
            the axes to plot in
        """
        # Get axes
        ax = ax or plt.gca()
        # Get the domain from the provided abscissas if not given
        start = start or np.min(self.abscissas)
        stop = stop or np.max(self.abscissas)

        # Evaluate the function and its derivative
        x = np.linspace(start, stop, num)
        if self.jac:
            y = np.asarray([self.fun(_x, *self.args) for _x in x])
        else:
            y = np.asarray([self.fun(_x, *self.args)[0] for _x in x])

        # Plot the function
        ax.plot(x, y - self._fun_maximum)
        # Mark the evaluation points
        ax.scatter(self.abscissas, self.fun_values)

        # Iterate over all points
        n = len(self.abscissas)
        for i in range(n):
            # Plot the upper hull
            lower = start if i == 0 else self.hull_abscissas[i]
            upper = stop if i == n - 1 else self.hull_abscissas[i + 1]
            jac_value = self.jac_values[i]
            fun_value = self.fun_values[i]
            abscissa = self.abscissas[i]
            ax.plot((lower, upper), (fun_value + jac_value * (lower - abscissa),
                                     fun_value + jac_value * (upper - abscissa)), color='r')

            # Plot the lower bound
            if i < n - 1:
                ax.plot((abscissa, self.abscissas[i + 1]), (fun_value, self.fun_values[i + 1]), color='g')

    def sample(self, size=None):
        """
        Draw a sample from the distribution.

        Parameters
        ----------
        size : int, optional
            number of samples to draw. Default is 1.

        Returns
        -------
        sample : float or array_like
            sample from the distribution or array of samples if `size` is given
        """
        if size is not None:
            return np.asarray([self.sample() for _ in range(size)])

        # Ensure that we have sensible starting values
        if self.jac_values[0] <= 0:
            raise ValueError('The smallest abscissa has non-positive gradient. Try decreasing it.')
        if self.jac_values[-1] >= 0:
            raise ValueError('The largest abscissa has non-negative gradient. Try increasing it.')

        while True:
            # Sum up the weights
            cumulative = np.cumsum(self.hull_weights)
            # Draw a uniform random number and identify into which bin it falls
            x = np.random.uniform(0, 1) * cumulative[-1]
            idx = bisect.bisect_left(cumulative, x)
            # Subtract the cumulative weight of all the previous segments
            if idx > 0:
                x -= cumulative[idx - 1]
            assert 0 < x < self.hull_weights[idx], "A hull segment was not correctly identified."
            # Apply the inverse transform to get a random variable from the upper hull
            lower = self.hull_abscissas[idx]
            jac_value = self.jac_values[idx]
            abscissa = self.abscissas[idx]
            fun_value = self.fun_values[idx]
            # Invert the cumulative distribution
            if np.isfinite(lower):
                sample = lower + np.log(1 + x * jac_value * np.exp(-fun_value + jac_value * (abscissa - lower))) / \
                                 jac_value
            else:
                # The inversion doesn't work if the lower limit is not finite
                sample = abscissa + (np.log(x * jac_value) - fun_value) / jac_value
            # Ensure the sample is in the right domain
            assert lower < sample < self.hull_abscissas[idx + 1], "A sample was drawn outside the identified segment."

            # Evaluate the squeezing function if the sample is to the left of the abscissa
            if sample < abscissa:
                # No lower bound defined if the sample lies to the left of all abscissas
                if idx == 0:
                    squeezing = -np.inf
                else:
                    # Linear interpolation with the point to the left of the abscissa
                    squeezing = fun_value + (fun_value - self.fun_values[idx - 1]) / \
                                            (abscissa - self.abscissas[idx - 1]) * (sample - self.abscissas[idx - 1])
            # Evaluate the squeezing function if the sample is to the right of the abscissa
            else:
                if idx == len(self.abscissas) - 1:
                    squeezing = -np.inf
                else:
                    # Linear interpolation with the point to the right of the abscissa
                    squeezing = fun_value + (self.fun_values[idx + 1] - fun_value) / \
                                            (self.abscissas[idx + 1] - abscissa) * (sample - abscissa)

            # Evaluate the upper hull
            hull = fun_value + jac_value * (sample - abscissa)

            # Accept or reject the proposal
            w = np.random.uniform(0, 1)
            if w <= np.exp(squeezing - hull):
                return sample

            # Add the point and accept or reject
            fun_value, jac_value = self.add_abscissa(sample)
            if w <= np.exp(fun_value - hull):
                return sample


def log_gaussian(x, mu=0.0, sigma=1.0):
    """
    Evaluate the log of the (unnormalised) Gaussian distribution.

    Parameters
    ----------
    x : float
        point at which to evaluate the distribution
    mu : float
        mean of the distribution
    sigma : float
        standard deviation of the distribution

    Returns
    -------
    value : float
        value of the log of the pdf
    jac : float
        derivative of the log of the pdf
    """
    value = -0.5 * (x - mu) ** 2 / sigma ** 2
    jac = (mu - x) / sigma ** 2
    return value + 10000, jac


def __main__():
    from scipy import stats
    np.random.seed(4)

    mu, sigma = 1.2, 1.5
    # Create an adaptive rejection sampler
    ars = AdaptiveRejectionSampler(log_gaussian, (-1, 2), (mu, sigma))

    # Draw 16 samples and add new abscissas in the process
    fig, axes = plt.subplots(4, 4, True, True)
    samples = []
    for ax in axes.ravel():
        ars.plot(ax=ax)
        ax.scatter(samples, np.zeros_like(samples), color='c')
        samples.append(ars.sample())

    # Summary of the data for a larger sample
    samples = ars.sample(100)
    print "Sample mean: {}".format(np.mean(samples))
    print "Sample std : {}".format(np.std(samples))

    # Try a hypothesis test to check that we have zero-mean data
    result = stats.ttest_1samp(samples, mu)
    print "p-value for sample mean t-test: {}".format(result.pvalue)

    plt.show()


if __name__ == '__main__':
    __main__()
