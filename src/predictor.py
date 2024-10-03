from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from kernel import Kernel


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


class KernelizedLinearPredictor(LinearPredictor):
    def __init__(self, kernel: Kernel, training_points: np.ndarray, training_labels: np.ndarray):
        training_size, _ = training_points.shape
        self.kernel = kernel
        self.training_points = training_points
        self.training_labels = training_labels
        self.alpha = np.zeros(training_size)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        k = self.kernel(self.training_points, X)
        d = np.dot(np.multiply(self.alpha, self.training_labels), k)
        return np.sign(d)

    def update(self, i):
        self.alpha[i] += 1
