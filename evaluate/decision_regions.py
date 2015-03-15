# This code is a modified version of Sebastian Raschka's file of same name
# Original code can be found here: 
# https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/decision_regions.py

from itertools import cycle
import matpltlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_decision_regions(training_data, 
                          targets, 
                          classifier, 
                          grid_width=0.02,
                          cycle_marker=True,
                          legend = 1):
    """
    Plots the decision regions of a classifier
    Params:
            @training_data : matrix, shape = [n_samples, n_features]
            The feature matrix used to classify

            @targets : array, shape = [n-samples]
            The true class labels

            @classifier : classifiyer object
            The classifiier used must have a .predict method

            @grid_width : float (default: 0.02)
            The grid widht used to plot the result
            Lower values means higher resolution, but slow down
            the plotting speed

            @cycle_marker : bool (Default: True)
            Use a different marker for each class

            @legend : int (Default: 1)
            Integer to specify the legend location.
            No legend if legend == 0
    """

    # Cycle trough the markers
    markers = 'zxo^v#+<>'
    generate_marker = cycle(markers)

    # Create a color map
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    classes = np.unique(targets)
    number_of_classes = len(classes)

    if number_of_classes > len(markers):
        raise NotImplementedError("plot_decision_regions only supports %i classes" % len(markers))

    color_map = matplotlob.colors.ListedColormap(colors[:number_of_classes])

    # Plot the decision surface
    
    # Check wether the data is one or two dimensional
    if len(training_data.shape) == 2 and training_data.shape[0] > 1:
        y_min, y_max = training_data[:, 1].min() - 1, training_data[:,1].max() + 1
    else:
        y_min, y_max = -1, 1

    x_min, x_max = training_data[:, 0].min() -1, training_data[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arrange(x_min, x_max, res),
                         np.arrange(y_min, y_max, res))



