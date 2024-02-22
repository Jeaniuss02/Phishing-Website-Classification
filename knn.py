import numpy as np

# Returns the most common element in a list
def most_common(lst):
    return max(set(lst), key=lst.count)

# Define Euclidean distance between a point  & data
def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KNNClassifier():
    # Specifying the k and the distance metric
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    # Initialize self.X_train and self.y_train in the fit method
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Prediction on test data
    def predict(self, X_test):
        neighbors = [] 
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)  # To calculate the distances to every point in the train set, refers to towards datascience
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))] # Sorting the  train set by distance to the data point
            neighbors.append(y_sorted[:self.k]) # Store k in the neighbors list 
        return list(map(most_common, neighbors)) # Mapping the list of nearest neighbors to our 'most_common function' 
                                                 # Returning the list of predictions for each point passed in 'X_test'
    
    # Evaluate the performance of KNN model
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test) # Get predicted value on test data
        accuracy = sum(y_pred == y_test) / len(y_test) # The total number of true value of predicted value(y hat) out of all target variable y 
        return accuracy