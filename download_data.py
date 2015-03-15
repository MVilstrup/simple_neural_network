import pandas as pd

# The data downloaded is the classical example of iris flower calssification
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Only use the setosa and versicolor as target data
target = data_file.iloc(0:100, 4].value

# Set setosa to be the class -1 and versicolor to be the class 1
taget = np.where(target == "Iris-setosa", -1, 1)

# Set the training data to be the sepal and petal length of the two kinds of iris
training_data = data_file.iloc[0:100, [0,2]].values


