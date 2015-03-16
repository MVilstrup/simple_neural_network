import numpy as np

class Adaline(object):
    """
    ADAptive LInear NEuron classifier
    Params:
            @learning_rate : float  (Learning rate between 0.0 and 1.0)
            @passes : int           (Number of passes over the training data)
            @random_state : int     (Random state for initializing weights)
            
            @learning_type : str    (Default: sgd)
            Gradient descent = "gd", Stochastic gradient descent = "sgd"
    
    Attributes:
            @ weights : 1D-array    (The weights after fitting)
            @ cost : 1D-array       (Sum of squared errors after each pass)
    """

    def __init__(self, 
                 learning_rate=0.01, 
                 passes=50, 
                 random_state=1, 
                 learning_type="sgd"):

        np.random.seed(random_state)
        self.learning_rate=learning_rate
        self.passes=passes

        if not learning_type in ("gd", "sgd"):
            raise ValueError("Learning type must be 'gd' or 'sgd'")
        self.learning_type=learning_type

    
    def fit(self, training_data, targets, init_weights=None):
        """
        Fit the weights to the training data
        Params:
                @training_data : matrix, shape = [n_samples, n_features]
                training vectors, where n_samples is the number of samples 
                and n_features is the number of features
                
                @targets : 1D-array, shape = [n_samples]
                Target values

                @init_weights : 1D-array, shape = [n_features + 1]
                Initial weights for the classifier. If None, weights are 
                initialized to 0.

        Returns:
                @Self : object
        """

        if not len(training_data.shape) == 2:
            raise ValueError("Training data must be a Matrix.")

        if not (np.unique(targets) == np.array([-1, 1])).all():
            raise ValueError("Supports only binary class labels -1 and 1")

        if not isinstance(init_weights, np.ndarray):
            self.weights = np.random.random(1 + training_data.shape[1])
        else:
            self.weights = init_weights

        self.cost = []

        for _ in range(self.passes):
            
            cost = 0.0
            if self.learning_type == "gd":
                # Fit to the data with gradient descent
                target_val = self.net_input(training_data)
                errors = (targets - target_val)
                self.weights[1:] += self.learning_rate * training_data.T.dot(errors)
                self.weights[0] += self.learning_rate * errors.sum()
                cost = (errors**2).sum() / 2.0

            elif self.learning_type == "sgd":
                # Fit to the data with Stochastic gradient descent
                for training_val, target_val in zip(training_data, targets):
                    dif_val = self.net_input(training_val)
                    error = (target_val - dif_val)
                    self.weights[1:] += self.learning_rate * training_val.dot(error)
                    self.weights[0] += self.learning_rate * error
                    cost += error**2 / 2.0
            self.cost.append(cost)

        return self
    
    def net_input(self, training_data):
        """ Net input function """
        return np.dot(training_data, self.weights[1:]) + self.weights[0]

    def activation(self, training_data):
        """ Activation function """
        return self.net_input(training_data)

    def predict(self, training_data):
        """
        Predict class labels for training data
        Params:
                @training_data : matrix, shape = [n_samples, n_features]
                training vectors, where n_samples is the number of samples 
                and n_features is the number of features
       
        Returns:
                @class : int    (Predicted class label)
        """
        return np.where(self.activation(training_data) >= 0.0, 1, -1)
