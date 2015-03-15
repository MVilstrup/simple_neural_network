# This code is a modified version of Sebastian Raschka's file of same name
# Original code can be found here: 
# https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/decision_regions.py

from itertools import cycle
import matplotlib
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
    generate_marker = cycle("sxo^v")

    # Create a color map
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    classes = np.unique(targets)
    number_of_classes = len(classes)

    if number_of_classes > len(colors):
        raise NotImplementedError("plot_decision_regions only supports %i classes" % len(markers))

    color_map = matplotlib.colors.ListedColormap(colors[:number_of_classes])

    # Plot the decision surface
    
    # Check wether the data is one or two dimensional
    if len(training_data.shape) == 2 and training_data.shape[0] > 1:
        y_min, y_max = training_data[:, 1].min() - 1, training_data[:,1].max() + 1
    else:
        y_min, y_max = -1, 1

    x_min, x_max = training_data[:, 0].min() -1, training_data[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_width),
                         np.arange(y_min, y_max, grid_width))
    
    # 2D
    if len(training_data.shape) == 2 and training_data.shape[1] > 1:
        y_min, y_max = training_data[:, 1].min() - 1, training_data[:, 1].max() + 1
        prediction_axis = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    # 1D
    else:
        y_min, y_max = -1, 1
        prediction_axis = classifier.predict(np.array([xx.ravel()]).T)

    prediction_axis = prediction_axis.reshape(xx.shape)
    plt.contourf(xx, yy, prediction_axis, alpha=0.4, cmap=color_map)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Plot class samples
    for c in np.unique(targets):
        
        if len(training_data.shape) == 2 and training_data.shape[1] > 1:
            dimensions = training_data[targets==c, 1]
        else:
            dimensions = [0 for i in training_data[targets==c]]

        plt.scatter(training_data[targets==c, 0],
                    dimensions,
                    alpha=0.8,
                    c=color_map(c),
                    marker=next(generate_marker),
                    label=c)

    if legend:
        plt.legend(loc=legend, fancybox=True, framealpha=0.5)
