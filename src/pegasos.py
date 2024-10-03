from __future__ import annotations

import random
from math import exp

import numpy as np

from kernel import Kernel
from predictor import KernelizedLinearPredictor, LinearPredictor, Predictor


# TODO: doc string
# TODO: try minibatch variant and write and compare results on the report (also for logistic)
def pegasos(training_points: np.ndarray, training_labels: np.ndarray, regularization_coefficent=0.1, rounds=1000) -> LinearPredictor:
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

# TODO(*): the typing lies: we say we return predictor but in reality we still returrning a closure fix it
def kernelized_pegasos(training_points: np.ndarray, training_labels: np.ndarray, kernel: Kernel, regularization_coefficent=0.1, rounds=1000) -> Predictor:
    samples, _ = training_points.shape
    predictor = KernelizedLinearPredictor(kernel, training_points, training_labels)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        learning_rate = 1 / (regularization_coefficent * t)
        # TODO: should we exclude somehow the alpha for the current index? 
        if y_it * learning_rate * np.dot(np.multiply(predictor.alpha, training_labels), kernel(training_points, x_it)) < 1:
            predictor.update(random_index)
    # TODO(*): we should generalize the predictor into a protocol or something like that and create subclass for linear and the two kernel algorithm
    return predictor

def sigmoid(z: float) -> float:
    """Computes the sigmoid function over a scalar `z`"""
    return 1 / (1 + exp(-z))

# TODO: DOC
def train_regularized_logistic_classification(training_points: np.ndarray, training_labels: np.ndarray, regularization_coefficent=0.1, rounds=1000) -> LinearPredictor:
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
