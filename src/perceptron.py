from __future__ import annotations

import numpy as np
from predictor import Predictor, LinearPredictor
from kernel import Kernel

class Perceptron(LinearPredictor):
    def __init__(self, features: np.ndarray):
        super().__init__(features)
        self.updates = 0

    def update(self, point: np.ndarray, label: float):
        self.updates += 1
        self.features += label * point
    
    @staticmethod
    def zero(dimension: int) -> 'Perceptron':
        """Return a new linear predictor initialized completly at zero"""
        return __class__(np.zeros(dimension))


class KernelizedPerceptron(Predictor):
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
