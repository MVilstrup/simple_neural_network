
import numpy as np

class Perceptron(object):
    """
    Perceptron classifier
    Params:
            @learning_rate : float  (between 0.0 and 1.0)
            @passes : int           (Passes over the training dataset)
            @random_state : int     (Random state for initializing random weights)
    Attributes:
            @weights : 1d-array     (Weights after fitting)
            @cost : list            (Number of misclassifications in every pass)
    """
    def __init__(self, learning_rate=0.01, passes=50, random_state=1):
        self.learning_rate = learning_rate
        self.passes = passes
        np.random.seed(random_state)
    
    def fit(self, training_data, targets, init_weights=None):
        """
        Fit training data
        Params:
                @training_data : sparse matrix, shape = [n_samples, n_features]
                (Training vectors, where n_samples is the number of samples 
                and n_features is the number of features.)
                @taget : array-like, shape = [n_samples] (Target values)
                
                @init_weights : array-like, shape = [n_features + 1]
                (Initial weights for the classifier. If None, weights are 
                initialized to 0.
        
        Returns: self : object
        """
        
        if not len(training_data.shape) == 2:
            raise ValueError("Training Data must be a 2D array. Try Training_data[:,np.newaxis]")
        
        if not (np.unique(targets) == np.array([-1, 1])).all():
            raise ValueError("Targets should only be binary class labels: -1 and 1")

        if not isinstance(init_weights, np.ndarray):
            self.weights = np.random.random(1 + training_data.shape[1])
        else:
            self.weights = init_weigths

        self.cost = []
        
        # Pass through the training data multiple times and fit the weights of the perceptron at each pass 
        for _ in range(self.passes):
            errors = 0
            for train_example, target in zip(training_data, targets):
                update = self.learning_rate * (target - self.predict(train_example))
                self.weights[1:] += update * train_example
                self.weights[0] += update
                errors += int(update != 0.0)
            self.cost.append(errors)

        return self

    def net_input(self, training_data):
        """ Net input function  """
        return np.dot(training_data, self.weights[1:]) + self.weights[0]

    def predict(self, training_data):
        """
        Predict class lables for Training Data.
        Params:
                @training_data : sparse matrix, shape = [n_samples, n_features]
                (Training vectors, where n_samples is the number of samples 
                and n_features is the number of features.)
        Returns:
                @class : int    (Predicted class label)
        """
        net_input = self.net_input(training_data)
        return np.where(net_input >= 0.0, 1, -1)

