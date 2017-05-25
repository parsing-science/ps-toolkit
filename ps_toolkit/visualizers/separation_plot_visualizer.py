from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError

from ps_toolkit.exc import PSToolkitError


class SeparationPlotVisualizer(object):
    """A class that can create a separation plot for a set of data with a binary outcome."""

    def separate_probabilities(self, probabilities, Y):
        """Function to separate the probabilities for events that are true and those that are false.
            Useful for creating a separation plot.

        Parameters
        ----------
        probabilities : a numpy array, list, or tuple of the probabilities of a True outcome for all data points

        Y : pandas Series or 1-column DataFrame, shape [n_samples]
            The outcome data used to separate the data points that have outcome=True from those of outcome=False

        """

        if type(Y)==pd.DataFrame and len(Y.columns) != 1:
            raise PSToolkitError("Y must be a one-column DataFrame or Series.")

        if len(Y) != len(probabilities):
            raise PSToolkitError("The probabilities and Y must be the same size.")

        if Y.isnull().any().any():
            raise PSToolkitError("Y contains NaNs.")

        if type(probabilities) != np.ndarray:
            probabilities = np.array(probabilities)

        if np.isnan(probabilities).any():
            raise PSToolkitError("The probabilities contains NaNs.")

        if not (probabilities >= 0).all() or not (probabilities <= 1).all():
            raise PSToolkitError("The probabilities must be between 0 and 1.")

        true_probs = []
        false_probs = []

        if type(Y)==pd.Series:
            Y = pd.DataFrame(Y)

        for i in range(len(Y)):
            if Y.iloc[i][0]:
                true_probs.append(probabilities[i])
            else:
                false_probs.append(probabilities[i])

        self.true_probs_ = true_probs
        self.false_probs_ = false_probs
     

    def create_separation_plot(self):
        """Function to create a separation plot for a set of true probabilities and false probabilities.
        """

        if not hasattr(self, "true_probs_") or not hasattr(self, "false_probs_"):
            raise NotFittedError("Call separate_probabilities before create_separation_plot")

        tints = [
            "#f3e6ed", 
            "#e7cedc", 
            "#dcb5ca", 
            "#d09db9", 
            "#c584a7", 
            "#b96c96", 
            "#ad5384", 
            "#a23a72",  
            "#962261", 
            "#8b0a50"
        ]

        a_heights, a_bins = np.histogram(self.true_probs_, 
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
 
        a_widths = a_heights/len(self.true_probs_)
     
        b_heights, b_bins = np.histogram(self.false_probs_, 
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
     
        b_widths = b_heights/len(self.false_probs_)

        plt.subplot(2, 1, 1)

        left_edge=0

        for i in range(10):
            plt.bar(left_edge, 
                1, 
                a_widths[i], 
                color=tints[i], 
                edgecolor=None, 
                label=str(i/10)+"-"+str((i+1)/10)
            )

            left_edge+=a_widths[i]
     
        plt.title("y=True (n={})".format(len(self.true_probs_)))

        plt.tick_params(axis='both',
            which='both', 
            left='off', 
            top='off', 
            bottom = 'off', 
            right='off', 
            labelleft='off', 
            labelbottom = 'off'
        )

        plt.legend(bbox_to_anchor=(1.05, 0.7), loc=2)   

        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,1,y1,y2))

        plt.subplot(2, 1, 2)

        left_edge=0

        for i in range(10):
            plt.bar(left_edge, 
                1, 
                b_widths[i], 
                color=tints[i], 
                edgecolor=None
            )

            left_edge+=b_widths[i]

        plt.title("y=False (n={})".format(len(self.false_probs_)))

        plt.tick_params(axis='both',
            which='both', 
            left='off', 
            top='off', 
            bottom = 'off', 
            right='off', 
            labelleft='off', 
            labelbottom = 'off')

        x1,x2,y1,y2 = plt.axis()
        plt.axis((0,1,y1,y2))

    def separate_and_plot(self, probabilities, Y):
        """A function that combines the functionality of _separate_probabilities and _create_separation_plot.

        Parameters
        ----------
        probabilities : a numpy array of the probabilities of a True outcome for all data points

        Y : pandas Series or 1-column DataFrame, shape [n_samples]
            The outcome data used to separate the data points that have outcome=True from those of outcome=False

        """

        self.separate_probabilities(probabilities, Y)

        self.create_separation_plot()
