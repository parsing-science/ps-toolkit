from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_curve, auc

from core.exc import PSToolkitError


class ROCCurveVisualizer(object):
    """" A class to create an ROC curve"""

    def calculate_roc(self, probabilities, Y, pos_label=None, sample_weight=None):
        """
        Function to calculate the ROC and AUC for a set of binary data. This function is built on top of the sklearn functions for those values.

        Parameters
        ----------
        probabilities : a numpy array, list, or tuple of the probabilities of a True outcome for all data points

        Y : pandas Series or 1-column DataFrame, shape [n_samples]
            The outcome data used to separate the data points that have outcome=True from those of outcome=False

        Optional from sklearn:
        pos_label : int, Label considered as positive and others are considered negative.
        
        sample_weight : array-like of shape = [n_samples], Sample weights.
        """

        try:
            probabilities = np.array(probabilities)
            probabilities = probabilities.squeeze()
        except:
            raise PSToolkitError("The probabilities must be castable to a numpy array.")

        if probabilities.ndim != 1:
            raise PSToolkitError("The probabilities must be a one dimensional numpy array, list, or tuple.")

        if type(Y)==pd.DataFrame and len(Y.columns) != 1:
            raise PSToolkitError("Y must be a one-column DataFrame or Series.")

        if len(Y) != len(probabilities):
            raise PSToolkitError("The probabilities and Y must be the same size.")

        if Y.isnull().any().any():
            raise PSToolkitError("Y contains NaNs.")

        if np.isnan(probabilities).any():
            raise PSToolkitError("The probabilities contains NaNs.")

        if not (probabilities >= 0).all() or not (probabilities <= 1).all():
            raise PSToolkitError("The probabilities must be between 0 and 1.")

        fpr, tpr, thresholds = roc_curve(Y, probabilities, pos_label, sample_weight)
        self.fpr_ = fpr
        self.tpr_ = tpr
        self.thresholds_ = thresholds

        roc_auc = auc(fpr, tpr)
        self.roc_auc_ = roc_auc

    def create_roc_curve_plot(self):
        """
        Function to plot the ROC curve for a set of binary data.
        """

        if not hasattr(self, "fpr_"):
            raise NotFittedError("Call calculate_roc before create_roc_curve_plot")

        plt.figure()
        plt.plot(self.fpr_, self.tpr_, label='ROC curve (area = %0.2f)' % self.roc_auc_)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()

    def calculate_and_plot(self, probabilities, Y, pos_label=None, sample_weight=None):
        """
        Function to calculate and plot the ROC curve for a set of data.

        Parameters
        ----------
        probabilities : a numpy array, list, or tuple of the probabilities of a True outcome for all data points

        Y : pandas Series or 1-column DataFrame, shape [n_samples]
            The outcome data used to separate the data points that have outcome=True from those of outcome=False

        Optional from sklearn:
        pos_label : int, Label considered as positive and others are considered negative.
        
        sample_weight : array-like of shape = [n_samples], Sample weights.
        """

        self.calculate_roc(probabilities, Y, pos_label, sample_weight)

        self.create_roc_curve_plot()
