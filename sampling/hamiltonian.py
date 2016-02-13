import numpy as np
from base import BaseSampler, normal_log_posterior, normal_log_posterior_jac


class HamiltonianSampler(BaseSampler):
    """
    Hamiltonian Monte Carlo sampler.

    Parameters
    ----------
    fun : callable
        function to compute the (unnormalised) log-PDF and its derivative if `jac` is not given
    args : tuple, optional
        extra arguments passed to the log-likelihood function
    jac : callable, optional
        function to compute the derivative of the log-PDF. If `jac` is not given, `fun` must return the function value
        and the derivative.
    mass : float or array_like, optional
        mass (vector) associated with each parameter or the variance of the momenta
    """
    def __init__(self, fun, args=None, parameter_names=None, jac=None, mass=1.0, epsilon=0.1, steps=10):
        super(HamiltonianSampler, self).__init__(fun, args, parameter_names)
        self.jac = jac
        self.mass = mass
        self.epsilon = epsilon
        self.steps = steps

    def sample(self, params, steps=1, callback=None, full=False):
        params = np.asarray(params, dtype=float)
        p = len(params)

        for step in range(steps):
            # Sample the momentum and compute the kinetic energy
            momentum = np.random.normal(0, 1, p) * np.sqrt(self.mass)
            kinetic = 0.5 * np.sum(momentum ** 2 / self.mass)

            # Evaluate the Jacobian
            if self.jac:
                jac = self.jac(params, *self.args)
                fun_value = self.fun(params, *self.args)
            else:
                fun_value, jac = self.fun(params, *self.args)

            # Evolve the variables
            fun_value_end = None
            params_end = params.copy()
            params_sequence = [params_end.copy()]
            for step in range(self.steps):
                # Make a half step for the leapfrog algorithm
                momentum += 0.5 * self.epsilon * jac
                # Update the position
                params_end += self.epsilon * momentum / self.mass
                params_sequence.append(params_end.copy())
                # Evaluate the Jacobian
                if self.jac:
                    jac = self.jac(params_end, *self.args)
                else:
                    fun_value_end, jac = self.fun(params_end, *self.args)
                # Make another half-step
                momentum += 0.5 * self.epsilon * jac

            # Evaluate the function value if necessary
            if fun_value_end is None:
                fun_value_end = self.fun(params_end, *self.args)

            # Evaluate the kinetic energy
            kinetic_end = 0.5 * np.sum(momentum ** 2 / self.mass)

            # Accept or reject the step
            if np.log(np.random.uniform()) < fun_value_end - fun_value + kinetic - kinetic_end:
                params = params_end
                fun_value = fun_value_end

            self._samples.append(params)
            self._fun_values.append(fun_value)

            if callback:
                callback(params)

        if not full:
            return params

        if steps == 1:
            return params, np.asarray(params_sequence)

        raise ValueError("`full=True` is only allowed for single steps")


def __main__():
    from matplotlib import pyplot as plt
    np.random.seed(1)

    mean = np.asarray([-1, 1, 3])
    parameter_names = [r'$\mu_{{{0}}}$'.format(i + 1) for i in range(len(mean))]

    # Initialise the adaptive metropolis sampler
    sampler = HamiltonianSampler(normal_log_posterior, (mean,), parameter_names, normal_log_posterior_jac)
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