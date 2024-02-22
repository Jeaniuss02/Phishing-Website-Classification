import numpy as np

class Perceptron:
    def __init__(self, learning_rate, num_epochs):  # initialize parameters 
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def fit(self, X, y):   # initialize weights and bias term to zero 
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_epochs):                              # for every epochs 
            for i in range(X.shape[0]):
                prediction = np.sign(np.dot(self.weights, X[i]) + self.bias) # compute activation, refers to github
                if prediction != y[i]:                                # the prediction is diffrent from label
                    self.weights += self.learning_rate * y[i] * X[i]  # update weight with learning rate
                    self.bias += self.learning_rate * y[i]            # update bias with learning rate

    def predict(self, X):  # make predictions for input
        return np.sign(np.dot(X, self.weights) + self.bias)