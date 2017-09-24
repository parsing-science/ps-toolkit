import joblib
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

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
            self.shared_vars[key].set_value(shared_vars[key])

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
        raise NotImplementedError

    def predict(self, X, *args):
        """
        Predicts labels of new data with a trained model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        args: tuple of optional arguments for the model
        """
        if args:
            ppc_mean = self.predict_proba(X, *args)
        else:
            ppc_mean = self.predict_proba(X)

        pred = ppc_mean > 0.5

        return pred

    def score(self, X, y, *args):
        """
        Scores new data with a trained model.

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        y : numpy array, shape [n_samples, ]

        args: tuple of optional arguments for the model
        """

        return accuracy_score(y, self.predict(X, *args))

    def save(self, file_prefix, custom_params=None):
        """
        Saves the advi_trace, v_params, and extra param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        path and prefix used to identify where to save trace and params for this model.
        ex. given file_prefix = "path/to/file/"
        This will attempt to save to "path/to/file/advi_trace.pickle" and "path/to/file/params.pickle"

        custom_params: Dictionary of custom parameters to save. Defaults to None
        """
        fileObject = open(file_prefix + 'advi_trace.pickle', 'w')
        joblib.dump(self.advi_trace, fileObject)
        fileObject.close()

        fileObject = open(file_prefix + 'v_params.pickle', 'w')
        joblib.dump(self.v_params, fileObject)
        fileObject.close()

        if custom_params:
            fileObject = open(file_prefix + 'params.pickle', 'w')
            joblib.dump(custom_params, fileObject)
            fileObject.close()

    def load(self, file_prefix, load_custom_params=False):
        """
        Loads a saved version of the advi_trace, v_params, and extra param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        path and prefix used to identify where to load saved trace and params for this model.
        ex. given file_prefix = "path/to/file/"
        This will attempt to load "path/to/file/advi_trace.pickle" and "path/to/file/params.pickle"

        load_custom_params: Boolean flag to indicate whether custom parameters should be loaded. Defaults to False.

        Returns
        ----------
        custom_params: Dictionary of custom parameters
        """
        self.advi_trace = joblib.load(file_prefix + 'advi_trace.pickle')
        self.v_params = joblib.load(file_prefix + 'v_params.pickle')

        custom_params = None
        if load_custom_params:
            custom_params = joblib.load(file_prefix + 'params.pickle')

        return custom_params

    def plot_elbo(self):
        """
        Plot the ELBO values after running ADVI minibatch.
        """
        sns.set_style("white")
        plt.plot(self.v_params.elbo_vals)
        plt.ylabel('ELBO')
        plt.xlabel('iteration')
        sns.despine()

    @staticmethod
    def _create_minibatch(data, num_samples, size=100):
        """
        Generator that returns mini-batches in each iteration
        """
        while True:
            # Return random data samples of set size each iteration
            ixs = np.random.randint(num_samples, size=size)
            yield [tensor[ixs] for tensor in data]
