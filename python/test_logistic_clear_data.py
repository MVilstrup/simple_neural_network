import pandas as pd
import matplotlib.pyplot as plt
from evaluate import plot_decision_regions
from time import sleep
import numpy as np
from classifier import LogisticRegression

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
training_data_std[:,1] /= training_data[:,1].std()


# Use Adaline with Gradient Descent
log_reg = LogisticRegression(learning_rate=0.01, 
        passes=15, 
        learning_type="gd",
        lambda_=1)
log_reg.fit(training_data_std, targets)

plot_decision_regions(training_data_std, targets, classifier=log_reg)
plt.title("Logistic Regression - Gradient Descent")
plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.show()

# Use Adaline with Stochastic gradient descent
log_reg = LogisticRegression(learning_rate=0.01, 
        passes=15, 
        learning_type="sgd",
        lambda_=0.6)
log_reg.fit(training_data_std, targets)

plot_decision_regions(training_data_std, targets, classifier=log_reg)
plt.title("Logistic Regression- Stochastic Gradient Descent")
plt.xlabel("Sepal length (standardized)")
plt.ylabel("Petal length (standardized)")
plt.show()
