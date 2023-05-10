from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np
from scipy.spatial.distance import cdist

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)
        return self
    
    def predict(self, X):
        # take distances with 2-norm from trained data
        Y = cdist(X, self.X, 'euclidean')
        # take n_neighbors
        idx = np.argpartition(Y, self.n_neighbors, axis=1)
        neighbors_idx = idx[:,:self.n_neighbors]

        # calculate mean of n_neighbors and take sign for prediction
        predictions = np.sign(np.mean(self.y[neighbors_idx], axis=1))
        return predictions