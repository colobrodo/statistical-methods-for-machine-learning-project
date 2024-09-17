from __future__ import annotations

from typing import Protocol
import numpy as np


class Kernel(Protocol):
    """Identifies a kernel, so a function that computes the dot product between the feature expansion of two vector"""
    def __call__(self, X: np.ndarray, X2: np.ndarray) -> float:
        """This method compute the kernel over the ndarray `X` and `X2`.
        They can be two vector and the kernel behave as mathematicly defined or they can be 2 matrix (`m x d`) and (`n x d`).   
        In the last case a new matrix `K` (`m x n`) is returned where in each position `K[i, j]` is stored the result of the kernel between `X[i]` and `X2[j]`
        """
        ...


class PolynomialKernel(Kernel):
    def __init__(self, degree: float) -> None:
        """A kernel that computes the dot product between the polynomial feature expansion of the two vector of degree `degree`

        :param degree: the degree of the polynomial in kernel space
        """
        self.degree = degree
    
    def __call__(self, X: np.ndarray, X2: np.ndarray):
        # I cannot find a way to use the same implementation for vector and matrices
        # in the case of vector I would like to use the simple dot product
        # for the matrix case instead I want to return the matrix K where K_i,j = K(x_i, x2_j)
        # mainly to avoid using python for loop and reduce the cost of the kernel evaluation on the whole training set
        if X2.ndim == 1:
            return np.power(np.dot(X, X2) + 1, self.degree)
        elif X2.ndim == 2:
            return np.power(np.dot(X, X2.T) + 1, self.degree)
    
    def __repr__(self):
        return f"PolynomialKernel(degree={self.degree})"


class GaussianKernel(Kernel):
    def __init__(self, gamma: float):
        """Creates a gaussian kernel of parameter `gamma`

        :param gamma: the gamma parameter, so the scaling factor of the distance between the two points"""
        self.gamma = gamma
    
    def __call__(self, X: np.ndarray, X2: np.ndarray):
        if X2.ndim == 1:
            dist = np.linalg.norm(X - X2, 2, axis=1)
        elif X2.ndim == 2:
            dist = np.linalg.norm(X[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)
        return np.exp(-dist / self.gamma)

    def __repr__(self):
        return f"GaussianKernel(gamma={self.gamma})"
