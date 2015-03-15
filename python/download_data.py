import pandas as pd
import matplotlib.pyplot as plt
from evaluate import plot_decision_regions
from time import sleep
import numpy as np
from perceptron import Perceptron

# The data downloaded is the classical example of iris flower calssification
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Only use the setosa and versicolor as target data
target = data_file.iloc[0:100, 4].values

# Set setosa to be the class -1 and versicolor to be the class 1
target = np.where(target == "Iris-setosa", -1, 1)

# Set the training data to be the sepal and petal length of the two kinds of iris
training_data = data_file.iloc[0:100, [0,2]].values

perceptron = Perceptron(passes=10, learning_rate=0.1)

perceptron.fit(training_data, target)
print "Weights: %s" % perceptron.weights

plot_decision_regions(training_data, target, classifier=perceptron)
plt.title("Perceptron")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.show()


plt.plot(range(1, len(perceptron.cost)+1), perceptron.cost, marker="o")
plt.xlabel("Iterations")
plt.ylabel("Missclassifications")
plt.show()

