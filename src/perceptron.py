from __future__ import annotations

import numpy as np
from predictor import Predictor, LinearPredictor
from kernel import Kernel

# TODO: DOC
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


# TODO: DOC
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


def train_perceptron(training_points: np.ndarray, training_labels: np.ndarray, max_epochs=10) -> Perceptron:
    """Given a training set returns a perceptron trained on it.
    Differently from the original algorithm the number of epoch is bounded to 
    a finite amount of epochs (`max_epochs`) and can terminate before convergence
    """
    _, d = training_points.shape
    perceptron = Perceptron.zero(d)
    for _ in range(max_epochs):
        update = False
        for x_t, y_t in zip(training_points, training_labels):
            y = perceptron.predict(x_t)
            if y_t * y <= 0:
                update = True
                perceptron.update(x_t, y_t)
        if not update:
            break
    return perceptron

def train_kernelized_perceptron(training_points: np.ndarray, training_labels: np.ndarray, kernel: Kernel, max_epochs=10) -> Perceptron:
    """Given a training set trains and returns a perceptron on the RKHS induced by the passed `kernel`.
    The algorithm terminates after `max_epochs` epochs or when it converges.
    """
    perceptron = KernelizedPerceptron(kernel, training_points, training_labels)
    for _ in range(max_epochs):
        update = False
        for t, z_t in enumerate(zip(training_points, training_labels)):
            x_t, y_t = z_t
            y = perceptron.predict(x_t)
            if y_t * y <= 0:
                update = True
                perceptron.update(t)
        if not update:
            break
    return perceptron