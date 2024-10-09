from __future__ import annotations

import numpy as np

from kernel import Kernel
from predictor import KernelizedLinearPredictor, LinearPredictor


class Perceptron(LinearPredictor):
    """This class represent an instance of the perceptron, with a `predict` 
    method.   
    It's also possible to update the weight of the underling linear predictor
    with a wrong predicted sample using the `update` procedure
    """
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
    perceptron = KernelizedLinearPredictor(kernel, training_points, training_labels)
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