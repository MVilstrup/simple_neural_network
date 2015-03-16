import pandas as pd
import matplotlib.pyplot as plt
from evaluate import plot_decision_regions
from time import sleep
import numpy as np
from classifier import Adaline

# The data downloaded is the classical example of iris flower calssification
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Only use the setosa and versicolor as target data
targets = data_file.iloc[0:100, 4].values

# Set setosa to be the class -1 and versicolor to be the class 1
targets = np.where(targets == "Iris-setosa", -1, 1)

# Set the training data to be the sepal and petal length of the two kinds of iris
training_data = data_file.iloc[0:100, [0,2]].values

# When using Gradient descent it is important to remember to standadize the data
# Standadization makes it much easier to find an appropriate learning rate
# and it often leads to faster convergence and van prevent weights from becomming
# too small
training_data_std = np.copy(training_data)

# Standardize rows
training_data_std[:,0] = (training_data[:,0] - training_data[:,0].mean()) 
training_data_std[:,0] /= training_data[:,0].std()

# Standardize columns
training_data_std[:,1] = (training_data[:,1] - training_data[:,1].mean()) 
trainint_data_std[:,1] /= training_data[:,1].std()


# Use Adaline with Gradient Descent
ada = Adaline(passes=15, learning_rate=0.01, learning_type="gd")

ada.fit(training_data_std, targets)

plot_decision_regions(training_data_std, targets, classifier=ada)
plt.title("Adaline - Gradient Descent")
plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.show()


plt.plot(range(1, len(ada.cost)+1), ada.cost, marker="o")
plt.xlabel("Iterations")
plt.ylabel("Missclassifications")
plt.show()

# Use Adaline with Stochastic gradient descent

ada = Adaline(passes=15, learning_rate=0.01, learning_type="sgd")

ada.fit(training_data_std, targets)

plot_decision_regions(training_data_std, targets, classifier=ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.show()


plt.plot(range(1, len(ada.cost)+1), ada.cost, marker="o")
plt.xlabel("Iterations")
plt.ylabel("Missclassifications")
plt.show()

