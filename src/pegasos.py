from __future__ import annotations

import random
from math import exp

import numpy as np

from kernel import Kernel
from predictor import KernelizedLinearPredictor, LinearPredictor, Predictor


def pegasos(training_points: np.ndarray, training_labels: np.ndarray, regularization_coefficent=0.1, rounds=1000) -> LinearPredictor:
    """This function trains and returns a linear predictor using the Pegasos algorithm with the 
    given training set passed in the first two parameters (`training_points` and 
    `training_labels`).    
    The regularization coefficent is passed in the optional parameter `regularization_coefficent`,
    and the number of rounds is instead choosed by the parameter with the same name (defaults to 
    `1000`).
    """
    samples, features = training_points.shape
    w = np.zeros(features)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        # choose the learning rate for this round 
        learning_rate = 1 / (t * regularization_coefficent)
        # update the predictor according to the hinge loss gradient 
        if y_it * np.dot(w, x_it) < 1:
            w = (1 - 1 / t) * w + learning_rate * y_it * x_it
        else:
            w = (1 - 1 / t) * w
    # REPORT:
    # NOTE: I adapted this version of the algorithm from https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf
    #       the name of the variable are adapted to be consistent with the pseudo code reported
    #       there are some differences with the pseudocode presented during the lecture:
    #       - the gradient descent update is written in a slightly different way using a conditional instead of an indicator function, I choose to remain consistent also with this stilistic choice
    #       - Instead of return the average of all the weight vector calculated at each step, the paper returns only the last one. The authors indicates that they notate an improvment in performance returning the last vector instead of the average
    #       - The author also provide a mini-batch version of the Pegasos algorithm, with another hyperparameter (the mini-batch size k)
    #       Another approach proposed by the paper is sampling without replacement: so a random permutation of the training set is choosen and the updates are performed in order on the new sequence of data.
    #       In this way in one epoch a training point is sampled only once. At the end of each epoch we can updated the predictor from the same permutation or shuffle the data another time.
    #       Although the authors report that this approach gives better results than uniform sampling as I did, I haven't experiment this variant of the algorithm
    return LinearPredictor(w)

def kernelized_pegasos(training_points: np.ndarray, training_labels: np.ndarray, kernel: Kernel, regularization_coefficent=0.1, rounds=1000) -> Predictor:
    """This function trains and returns a linear predictor using the kernelized version of the 
    Pegasos algorithm with the given training set passed in the first two parameters 
    (`training_points` and `training_labels`).    
    The regularization coefficent is passed in the optional parameter `regularization_coefficent`,
    and the number of rounds is instead choosed by the parameter with the same name (defaults to 
    `1000`).
    A valid implementation of `Kernel` should be passed to the `kernel` parameter.
    """
    samples, _ = training_points.shape
    predictor = KernelizedLinearPredictor(kernel, training_points, training_labels)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        learning_rate = 1 / (regularization_coefficent * t)
        if y_it * learning_rate * np.dot(np.multiply(predictor.alpha, training_labels), kernel(training_points, x_it)) < 1:
            predictor.update(random_index)
    return predictor

def sigmoid(z: float) -> float:
    """Computes the sigmoid function over a scalar `z`"""
    return 1 / (1 + exp(-z))

def train_regularized_logistic_classification(training_points: np.ndarray, training_labels: np.ndarray, regularization_coefficent=0.1, rounds=1000) -> LinearPredictor:
    """This function trains and returns a linear predictor using the Logistic Regression algorithm.     
    The training set is passed in the first two parameters (`training_points` and
    `training_labels`).    
    The regularization coefficent is passed in the optional parameter `regularization_coefficent`,
    and the number of rounds is instead choosed by the parameter with the same name (defaults to 
    `1000`).
    """
    samples, features = training_points.shape
    w = np.zeros(features)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        # choose the learning rate for this round 
        learning_rate = 1 / (t * regularization_coefficent)
        # update the predictor according to the logistic loss gradient 
        w = (1 - 1 / t) * w + learning_rate * sigmoid(-y_it * np.dot(w, x_it)) * y_it * x_it        
    return LinearPredictor(w)
