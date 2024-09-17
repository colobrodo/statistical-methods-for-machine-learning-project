from __future__ import annotations


from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, X: np.ndarray):
        return self.predict(X)


class LinearPredictor(Predictor):
    """Represent a linear predictor that separate the space into two semi-space: one positive one negative"""
    def __init__(self, features: np.ndarray):
        self.features = features
        dimension = self.features.shape
        self.dimension = dimension
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given an array of shape (m x d) where d is the dimension of the 
        predictor, returns an array of m elements either -1 or 1 based on the
        sign of the dot product with the feature vector."""
        return np.sign(np.dot(X, self.features))
