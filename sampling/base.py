import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd


class ReportCallback(object):
    def __init__(self, frequency, *args):
        self.frequency = frequency
        self.current = 0
        self.args = args

    def __call__(self, parameters):
        self.current += 1
        if self.current % self.frequency == 0:
            print "{}: {}".format(self.current, parameters)
        # Call all the other callbacks
        for arg in self.args:
            arg(parameters)


class BaseSampler(object):
    def __init__(self, fun, args=None, parameter_names=None, break_on_interrupt=True):
        if not callable(fun):
            raise ValueError("`fun` must be callable")

        self.fun = fun
        self.args = [] if args is None else args
        self.parameter_names = parameter_names
        self.break_on_interrupt = break_on_interrupt

        self._samples = []
        self._fun_values = []

    def get_parameter_name(self, index):
        return str(index) if self.parameter_names is None else self.parameter_names[index]

    def trace_plot(self, burn_in=0, parameters=None, values=None):
        samples = self.samples[burn_in:]

        if parameters is None:
            parameters = np.arange(samples.shape[1])

        fig, (ax1, ax2) = plt.subplots(1, 2, True)

        for parameter in parameters:
            ax1.plot(samples[:, parameter], label=self.get_parameter_name(parameter))

        # Plot true values
        if values is not None:
            for value in values:
                ax1.axhline(value, ls='dotted')

        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Parameter values')
        ax1.legend(loc=0, frameon=False)

        ax2.plot(self.fun_values[burn_in:])
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Function values')

        fig.tight_layout()

        return fig, (ax1, ax2)

    def density_plot(self, burn_in=0, parameters=None, values=None, nrows=None, ncols=None, bins=10):
        samples = self.samples[burn_in:]

        if parameters is None:
            parameters = np.arange(samples.shape[1])

        # Determine the number of rows and columns if not specified
        n = len(parameters)
        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(float(n) / ncols))
        elif nrows is None:
            nrows = int(np.ceil(float(n) / ncols))
        elif ncols is None:
            ncols = int(np.ceil(float(n) / nrows))

        fig, axes = plt.subplots(nrows, ncols)

        # Plot all parameters
        for parameter, ax in zip(parameters, np.ravel(axes)):
            x = samples[:, parameter]
            ax.hist(x, bins, normed=True, histtype='stepfilled', facecolor='silver')

            min_x, max_x = np.min(x), np.max(x)
            rng_x = max_x - min_x
            lin_x = np.linspace(min_x - 0.1 * rng_x, max_x + 0.1 * rng_x)
            kde = gaussian_kde(x)
            ax.plot(lin_x, kde(lin_x), color='blue')
            ax.set_title(self.get_parameter_name(parameter))

        # Plot true values
        if values is not None:
            for value, ax in zip(values, np.ravel(axes)):
                ax.axvline(value, ls='dotted')

        fig.tight_layout()

        return fig, axes

    def acceptance_rate(self, burn_in=0):
        samples = self.samples[burn_in:]
        return np.mean(samples[1:] != samples[:-1])

    def sample(self, parameters, steps=1, callback=None):
        raise NotImplementedError

    def describe(self, burn_in=0, parameters=None, do_print=True):
        samples = self.samples[burn_in:]
        if parameters is None:
            parameters = np.arange(samples.shape[1])

        # Use pandas to get a description
        columns = map(self.get_parameter_name, parameters)
        frame = pd.DataFrame(samples, columns=columns)
        description = frame.describe()

        name = self.__class__.__name__

        description = "{}\n{}\n{}".format(name, '=' * len(name), description)

        if do_print:
            print description

        return description

    @property
    def samples(self):
        return np.asarray(self._samples)

    @property
    def fun_values(self):
        return np.asarray(self._fun_values)


def normal_log_posterior(parameters, mean=0, variance=1):
    return -0.5 * np.sum((parameters - mean) ** 2 / variance)


def normal_log_posterior_jac(parameters, mean=0, variance=1):
    return - (parameters - mean) / variance
