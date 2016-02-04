import numpy as np
import bisect
from matplotlib import pyplot as plt


class AdaptiveRejectionSampler:
    def __init__(self, fun, x0, args=None, jac=None, domain=(-np.inf, np.inf)):
        # Copy the function and the jacobian
        self.fun = fun
        self.jac = jac
        self.domain = domain
        # Store the extra arguments for the function
        self.args = [] if args is None else args
        # Create containers for the abscissas, function values and derivatives
        self.abscissas = []
        self.fun_values = []
        self.jac_values = []
        # Insert the domain values into the intersection points
        self.hull_abscissas = None
        # The weight of each segment
        self.hull_weights = []
        # Add the initial values
        self.add_abscissas(x0)
        # Ensure that we have sensible starting values
        if self.jac_values[0] <= 0:
            raise ValueError('The smallest abscissa has non-positive gradient. Try decreasing it.')
        if self.jac_values[-1] >= 0:
            raise ValueError('The largest abscissa has non-negative gradient. Try increasing it.')

    def _hull_intersection(self, idx):
        intersection = (self.fun_values[idx + 1] - self.fun_values[idx] - self.abscissas[idx + 1] *
                        self.jac_values[idx + 1] + self.abscissas[idx] * self.jac_values[idx]) / \
                       (self.jac_values[idx] - self.jac_values[idx + 1])
        assert self.abscissas[idx] < intersection < self.abscissas[idx + 1]
        return intersection

    def _update_hull_weight(self, idx):
        # Compute the integral of the tangent associated with the point at `idx`
        fun_value = self.fun_values[idx]
        jac_value = self.jac_values[idx]
        lower = self.hull_abscissas[idx]
        upper = self.hull_abscissas[idx + 1]
        abscissa = self.abscissas[idx]
        weight = np.exp(fun_value - jac_value * abscissa) * (np.exp(upper * jac_value) - np.exp(lower * jac_value)) / \
               jac_value

        self.hull_weights[idx] = weight

    def add_abscissa(self, abscissa):
        # Check that the abscissa is not outside the domain
        if abscissa < self.domain[0] or abscissa > self.domain[1]:
            raise ValueError("The abscissa {0} falls must be in the domain [{1}, {2}].".format(
                abscissa, *self.domain
            ))

        # Identify where the abscissa should be placed
        idx = bisect.bisect_left(self.abscissas, abscissa)
        # Insert the abscissa
        self.abscissas.insert(idx, abscissa)

        # Evaluate the function
        if self.jac:
            fun_value = self.fun(abscissa, *self.args)
            jac_value = self.jac(abscissa, *self.args)
        else:
            fun_value, jac_value = self.fun(abscissa, *self.args)

        # Add the function values
        self.fun_values.insert(idx, fun_value)
        self.jac_values.insert(idx, jac_value)

        # Do not update intersection points if we only have one abscissa
        if len(self.abscissas) < 2:
            return fun_value, jac_value
        # Initialise the hull intersection points
        elif len(self.abscissas) == 2:
            self.hull_abscissas = list(self.domain)
            self.hull_weights = range(1)

        # Remove the hull abscissa between adjacent points
        if 0 < idx < len(self.abscissas) - 1:
            del self.hull_abscissas[idx]

        # If the abscissa is not the largest one, compute the intersection with the next point
        if idx < len(self.abscissas) - 1:
            intersection = self._hull_intersection(idx)
            self.hull_abscissas.insert(idx or 1, intersection)

        # If the abscissa is not the smallest one, compute the intersection with the previous point
        if idx > 0:
            intersection = self._hull_intersection(idx - 1)
            self.hull_abscissas.insert(idx, intersection)

        # Insert a dummy weight for this segment
        self.hull_weights.insert(idx, None)

        # If this is not the smallest one, update the previous weight
        if idx > 0:
            self._update_hull_weight(idx - 1)

        self._update_hull_weight(idx)

        # If this is not the largest one, update the next weight
        if idx < len(self.abscissas) - 1:
            self._update_hull_weight(idx + 1)

        return fun_value, jac_value

    def add_abscissas(self, abscissas):
        for abscissa in abscissas:
            self.add_abscissa(abscissa)

    def plot(self, start=None, stop=None, num=50, ax=None):
        # Get axes
        ax = ax or plt.gca()
        # Get the domain from the provided abscissas if not given
        start = start or np.min(self.abscissas)
        stop = stop or np.max(self.abscissas)

        # Evaluate the function
        x = np.linspace(start, stop, num)
        if self.jac:
            y = self.fun(x, *self.args)
        else:
            y, _ = self.fun(x, *self.args)

        # Plot the function
        ax.plot(x, y)
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
        if size is not None:
            return [self.sample() for _ in range(size)]

        while True:
            # Sum up the weights
            cumulative = np.cumsum(self.hull_weights)
            # Draw a uniform random number and identify into which bin it falls
            x = np.random.uniform(0, 1) * cumulative[-1]
            idx = bisect.bisect_left(cumulative, x)
            # Subtract the cumulative weight of all the previous segments
            if idx > 0:
                x -= cumulative[idx - 1]
            assert 0 < x < self.hull_weights[idx]
            # Apply the inverse transform to get a random variable from the upper hull
            lower = self.hull_abscissas[idx]
            jac_value = self.jac_values[idx]
            abscissa = self.abscissas[idx]
            fun_value = self.fun_values[idx]
            # If the lower bound is finite
            if np.isfinite(lower):
                sample = lower + np.log(1 + x * jac_value * np.exp(-fun_value + jac_value * (abscissa - lower))) / jac_value
            else:
                sample = abscissa + (np.log(x * jac_value) - fun_value) / jac_value
            # Ensure the sample is in the right domain
            assert lower < sample
            assert sample < self.hull_abscissas[idx + 1]

            # Evaluate the squeezing function
            if sample < abscissa:
                if idx == 0:
                    squeezing = -np.inf
                else:
                    squeezing = fun_value + (fun_value - self.fun_values[idx - 1]) / (abscissa - self.abscissas[idx - 1]) * \
                                            (sample - self.abscissas[idx - 1])
            else:
                if idx == len(self.abscissas) - 1:
                    squeezing = -np.inf
                else:
                    squeezing = fun_value + (self.fun_values[idx + 1] - fun_value) / (self.abscissas[idx + 1] - abscissa) * \
                                            (sample - abscissa)

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

    def validate(self):
        assert len(self.abscissas) == len(self.fun_values), "Number of abscissas and function values do not match."
        assert len(self.abscissas) == len(self.jac_values), "Number of abscissas and Jacobian values do not match."
        assert len(self.abscissas) == len(self.hull_weights), "Number of abscissas and hull weights do not match."
        assert len(self.abscissas) == len(self.hull_abscissas) - 1, "Number of abscissas and hull abscissas do not match."


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
    return value, jac


if __name__=='__main__':
    from scipy.stats import norm

    np.random.seed(3)
    # Create an adaptive rejection sampler
    ars = AdaptiveRejectionSampler(log_gaussian, (-1, 1))

    fig, axes = plt.subplots(4, 4, True, True)
    samples = []
    for ax in axes.ravel():
        ars.plot(ax=ax)
        ax.scatter(samples, np.zeros_like(samples), color='c')
        samples.append(ars.sample())

    plt.show()

    '''
    ars.add_abscissas(np.random.uniform(-1, 1, 5))
    ars.validate()

    # Compare the normalised hull weights with the exact value
    print "Approximation"
    print np.cumsum(ars.hull_weights) / np.sum(ars.hull_weights)
    print "Exact"
    cdf = norm.cdf(ars.hull_abscissas[1:])
    print cdf

    ars.sample()

    # Show the envelope
    ars.plot(-1.1, 1.1)
    plt.show()
    '''