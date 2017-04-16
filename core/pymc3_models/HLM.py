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

from core.exc import PSToolkitError


class HLM(BaseEstimator):
    """
    Custom Hierachical Linear Model built using PyMC3.
    """

    def __init__(self):
        self.cached_model = None
        self.shared_vars = None
        self.num_cats = None
        self.num_pred = None
        self.advi_trace = None
        self.v_params = None

    def create_model(self):
        """
        Creates and returns the PyMC3 model.

        Returns the model and the output variable. The latter is for use in ADVI minibatch.
        """
        model_input = theano.shared(np.zeros([1, self.num_pred]))

        model_output = theano.shared(np.zeros(1))

        model_cats = theano.shared(np.zeros(1, dtype='int'))

        self.shared_vars = {'model_input': model_input, 'model_output': model_output, 'model_cats': model_cats}

        model = pm.Model()

        with model:
            # Both alpha and beta are drawn from Normal distributions
            mu_alpha = pm.Normal("mu_alpha", mu=0, sd=10)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sd=10)

            mu_beta = pm.Normal("mu_beta", mu=0, sd=10)
            sigma_beta = pm.HalfNormal("sigma_beta", sd=10)

            alpha = pm.Normal('alpha', mu=mu_alpha, sd=sigma_alpha, shape=(self.num_markets,))
            beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=(self.num_cats, self.num_pred))

            c = model_cats

            temp = alpha[c] + T.sum(beta[c] * model_input, 1)

            p = pm.invlogit(temp)

            o = pm.Bernoulli('o', p, observed=model_output)

        return model, o

    def _set_shared_vars(self, model_input, model_output, model_cats):
        """
        Creates theano shared variables for the PyMC3 model.
        """
        self.shared_vars['model_input'].set_value(model_input)

        self.shared_vars['model_output'].set_value(model_output)

        self.shared_vars['model_cats'].set_value(model_cats)

    def _inference(self, minibatch_tensors, minibatch_RVs, minibatches, num_samples):
        """
        Runs variational ADVI in minibatch form and then samples from those results.
        """
        with self.cached_model:
            v_params = pm.variational.advi_minibatch(
                n=50000,
                minibatch_tensors=minibatch_tensors,
                minibatch_RVs=minibatch_RVs,
                minibatches=minibatches,
                total_size=num_samples,
                learning_rate=1e-2,
                epsilon=1.0
            )

            advi_trace = pm.variational.sample_vp(v_params, draws=7500)

        return v_params, advi_trace

    def fit(self, X, cats, y):
        """
        Train the HLM model

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats: numpy array, shape [n_samples, ]

        y : numpy array, shape [n_samples, ]
        """
        self.num_cats = len(np.unique(cats))
        num_samples, self.num_pred = X.shape

        #model_input, model_output, model_markets = self._create_shared_vars(X, y, markets)

        if self.cached_model is None:
            self.cached_model, o = self.create_model()

        minibatch_tensors = [
            self.shared_vars['model_input'],
            self.shared_vars['model_output'],
            self.shared_vars['model_cats']
        ]
        minibatch_RVs = [o]

        minibatches = self._create_minibatch([X, y, cats], num_samples)

        self.v_params, self.advi_trace = self._inference(
            minibatch_tensors,
            minibatch_RVs,
            minibatches,
            num_samples
        )

        return self

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

    def predict_proba(self, X, markets):
        """
        Predicts probabilities of new data with a trained HLM

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        markets: numpy array, shape [n_samples, ]
        """

        if self.advi_trace is None:
            raise PSToolkitError("Run fit on the model before predict.")

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model, o = self.create_model()

        self._set_shared_vars(X, np.zeros(num_samples), cats)

        ppc = pm.sample_ppc(self.advi_trace, model=self.cached_model, samples=2000)

        return ppc['o'].mean(axis=0)

    def predict(self, X, cats):
        """
        Predicts labels of new data with a trained HLM

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats: numpy array, shape [n_samples, ]
        """
        ppc_mean = self.predict_proba(X, cats)

        pred = ppc_mean > 0.5

        return pred

    def score(self, X, cats, y):
        """
        Scores new data with a trained model.

        Parameters
        ----------
        X : numpy array, shape [n_samples, n_features]

        cats: numpy array, shape [n_samples, ]

        y : numpy array, shape [n_samples, ]
        """
        return accuracy_score(y, self.predict(X, cats))

    def save(self, file_prefix):
        """
        Saves the advi_trace, v_params, and param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        path and prefix used to identify where to save trace and params for this model.
        ex. given file_prefix = "path/to/file/"
        This will attempt to save to "path/to/file/advi_trace.pickle" and "path/to/file/params.pickle"
        """
        fileObject = open(file_prefix + "advi_trace.pickle", 'w')
        joblib.dump(self.advi_trace, fileObject)
        fileObject.close()

        fileObject = open(file_prefix + "v_params.pickle", 'w')
        joblib.dump(self.v_params, fileObject)
        fileObject.close()

        fileObject = open(file_prefix + "params.pickle", 'w')
        joblib.dump(
            {"num_cats": self.num_cats, "num_pred": self.num_pred},
            fileObject
        )
        fileObject.close()

    def load(self, file_prefix):
        """
        Loads a saved version of the advi_trace, v_params, and param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        path and prefix used to identify where to load saved trace and params for this model.
        ex. given file_prefix = "path/to/file/"
        This will attempt to load "path/to/file/advi_trace.pickle" and "path/to/file/params.pickle"
        """
        self.advi_trace = joblib.load(file_prefix + "advi_trace.pickle")
        self.v_params = joblib.load(file_prefix + "v_params.pickle")

        params = joblib.load(file_prefix + "params.pickle")
        self.num_cats = params["num_cats"]
        self.num_pred = params["num_pred"]
