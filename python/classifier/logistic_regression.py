import numpy as np

class LogisticRegression(object):
    """
    Logistic Regression Classifier

    Params:
            @learning_rate : float  (Learning rate between 0.0 and 1.0)
            @passes : int           (Number of passes over the training data)
            
            @learning_type : str    (Default: sgd)
            Learning type where:    "sgd" = Stochastic Gradiant Descent
                                    "gd"  = Gradient Descent

            @lambda_ : float
            regularization parameter for L2 regularization
            No regularization if lamda = 0.0


    Attributes:
            @weights : array        (Weights after fitting)
            @cost : array           (Sum of squared error after each pass)
    """

    def __init__(self, learning_rate=0.01, passes=50, lambda_=0.0, learning_type="sgd"):
        self.learning_rate=learning_rate
        self.passes=passes
        self.lambda_=lambda_

        if not learning_type in ("sgd", "gd"):
            raise ValueError("Learning must be 'sgd' or 'gd'")
        self.learning_type=learning_type

    def fit(self, training_data, targets, init_weights=None):
        """
        Fit weights to training data
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
            
            if self.learning_type == "gd":
                # Fit to data with Gradient Descent
                target_val = self.activate(training_data)
                errors = targets - target_val
                regularize = self.lambda_ * self.weights[1:]
                self.weights[1:] += self.learning_rate * training_data.T.dot(errors)
                self.weights[1:] += regularize
                self.weights[0] += self.learning_rate * errors.sum()
            
            elif self.learning_type == "sgd":
                # Fit to data with Stochastic Gradient Descent
                for training_val, target_val in zip(training_data, targets):
                    diff_val = self.activate(training_data)
                    error = (target_val - diff_val)
                    regularize = self.lambda_ * self.weights[1:]
                    self.weights[1:] += self.learning_rate * training_val.dot(error)
                    self.weights[1:] += regularize
                    self.weights[0] += self.learning_rate * error
            self.cost.append(self._logit_cost(targets, self.activate(training_data)))
        
        return self

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
        return np.where(self.activate(training_data) >= 0.5, 1 , 0)

    def activate(self, training_data):
        """ Activate function using a sigmoid"""
        z = training_data.dot(self.weights[1:]) + self.weights[0]
        return self_sigmoid(z)

    def _logit_cost(self, targets, target):
        """ Caluclate the logistic cost """
        logit = -targets.dot(np.log(target)) - ((1 - targets).dot(np.log(1 - target)))
        regularize = (self.lamda_ / 2) * self.weights[1:].dot(self.weights[1:])
        return logit + regularize

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

