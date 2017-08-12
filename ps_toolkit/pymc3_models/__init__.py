import joblib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
from six.moves import zip
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import theano
import theano.tensor as T

from ps_toolkit.exc import PSToolkitError


class BayesianModel(BaseEstimator):
    """
    Bayesian model base class
    """
    def __init__(self):
        self.cached_model = None
        self.shared_vars = None
        self.num_pred = None
        self.advi_trace = None
        self.v_params = None

    def create_model(self):
        raise NotImplementedError

    def _set_shared_vars(self, shared_vars):
        """
        Sets theano shared variables for the PyMC3 model.
        """
        for key in shared_vars.keys():
            self.shared_vars['key'].set_value(shared_vars[key])

    def _inference(self, minibatch_tensors, minibatch_RVs, minibatches, num_samples):
        """
        Runs minibatch variational ADVI and then sample from those results.
        """
        with self.cached_model:
            v_params = pm.variational.advi_minibatch(
                n=50000,
                minibatch_tensors=minibatch_tensors,
                minibatch_RVs=minibatch_RVs,
                minibatches=minibatches,
                total_size=int(num_samples),
                learning_rate=1e-2,
                epsilon=1.0
            )

            advi_trace = pm.variational.sample_vp(v_params, draws=7500)

        return v_params, advi_trace

    def fit(self):
        raise NotImplementedError

    def predict_proba(self):
        pass

    def predict(self):
        pass

    def score(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    @staticmethod
    def _create_minibatch(data, num_samples, size=100):
        """
        Generator that returns mini-batches in each iteration
        """
        while True:
            # Return random data samples of set size each iteration
            ixs = np.random.randint(num_samples, size=size)
            yield [tensor[ixs] for tensor in data]

    def plot_elbo(self):
        """
        Plot the ELBO values after running ADVI minibatch.
        """
        sns.set_style("white")
        plt.plot(self.v_params.elbo_vals)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        sns.despine()
