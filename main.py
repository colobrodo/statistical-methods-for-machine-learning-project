from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations_with_replacement, product
from functools import partial
from typing import Protocol, Any
from math import inf, log, exp

import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: create abc for predictors, losses and kernels and modules
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


class Perceptron(LinearPredictor):
    def __init__(self, features: np.ndarray):
        super().__init__(features)
        self.updates = 0

    def update(self, point: np.ndarray, label: float):
        self.updates += 1
        self.features += label * point
    
    @staticmethod
    def zero(dimension: int) -> Perceptron:
        """Return a new linear predictor initialized completly at zero"""
        return __class__(np.zeros(dimension))


def split_train_test_set(dataset: np.ndarray, training_size: float) -> tuple[np.ndarray, np.ndarray]:
    m, _ = dataset.shape
    training_size = int(m * training_size)
    return dataset[:training_size], dataset[training_size:] 

def standardise(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    X_val_norm = (X_val - mean) / std
    return X_train_norm, X_val_norm, X_test_norm

def scale(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    min_ = np.min(X_train, axis=0)
    max_ = np.max(X_train, axis=0)
    X_train_scaled = (X_train - min_) / (max_ - min_)
    X_test_scaled = (X_test - min_) / (max_ - min_)
    X_val_scaled = (X_val - min_) / (max_ - min_)
    return X_train_scaled, X_val_scaled, X_test_scaled

def zero_one_loss(labels: np.ndarray, predictions: np.ndarray) -> float:
    return np.not_equal(labels, predictions).astype(int)

def set_error(loss, predictor: Predictor, points: np.ndarray, labels: np.ndarray) -> float:
    # set size
    m = len(labels)
    predictions = predictor(points)
    return np.sum(loss(labels, predictions)) / m

def train_perceptron(training_points: np.ndarray, training_labels: np.ndarray, max_epochs=10) -> Perceptron:
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

# TODO: the number of rounds should be calculated and derivated (show it on the report) and not passed as hyperparameter
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

def kernelized_pegasos(training_points: np.ndarray, training_labels: np.ndarray, kernel: Kernel, regularization_coefficent=0.1, rounds=1000) -> np.ndarray:
    samples, features = training_points.shape
    alpha = np.zeros(samples)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        learning_rate = 1 / (regularization_coefficent * t)
        # TODO: should we exclude somehow the alpha for the current index? 
        if y_it * learning_rate * np.dot(alpha, np.apply_along_axis(lambda x: kernel(x, x_it), 1, training_points)) < 1:
            alpha[random_index] += 1
    # TODO: we should generalize the predictor into a protocol or something like that and create subclass for linear and the two kernel algorithm
    return alpha

def sigmoid(z: float) -> float:
    return 1 / (1 + exp(-z))

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

class HyperparameterSearchResult:
    def __init__(self, configuration: dict[str, Any], predictor: Predictor, objective: float) -> None:
        self.parameters = configuration
        self.predictor = predictor
        self.objective = objective

def grid_search(algorithm, training_points, training_labels, objective_function, **hyperparameters) -> HyperparameterSearchResult:
    best_objective = inf
    best_configuration = None
    best_predictor = None
    # TODO: CLEANUP
    for parameter_values in product(*hyperparameters.values()):
        # NOTE: from python version 3.7 the dictionary are garanteed to preserve insersion order during iteration
        #       for this reason we can iterate this using zip
        hyperparameters_configuration = dict(zip(hyperparameters, parameter_values))
        predictor = algorithm(training_points, training_labels, **hyperparameters_configuration)
        objective = objective_function(predictor)
        # DEBUG:
        # TODO: instead of this print we should provide an hook like function that accept all the current hyper parameter configuration
        #       with objective (validation error) and maybe training error 
        #       in this way we can fix all the hyperparameters and let only one varying, collect all the point for that hyperparameter and
        #       plot at the end of the function using matplotlib
        #       
        #       plotter = HyperparameterPlotter(hyperparam='regularization_coefficent')
        #       grid_search(..., plotter.add_point)
        #       plotter.plot_validation_error()
        #       plotter.plot_test_error()
        #       plotter.show()
        print(f"{hyperparameters_configuration} -> {objective}")
        if objective < best_objective:
            best_objective = objective
            best_configuration = hyperparameters_configuration
            best_predictor = predictor

    print("-" * 50 + '\n')
    return HyperparameterSearchResult(best_configuration, best_predictor, best_objective)

def polynomial_feature_expansion(X: np.ndarray, degree: int) -> np.ndarray:
    samples, features = X.shape
    # Generate combinations of feature indices up to the given degree
    combinations = []
    for d in range(1, degree + 1):
        combinations.extend(combinations_with_replacement(range(features), d))
    # TODO:
    # Create a list to hold the new features
    poly_features = np.empty((samples, len(combinations)), dtype=X.dtype)
    # Generate the new features
    for i, comb in enumerate(combinations):
        poly_features[:, i] = np.prod(X[:, comb], axis=1)
    return poly_features

class Kernel(Protocol):
    def __call__(self, X: np.ndarray, X2: np.ndarray) -> float:
        ...


def create_polynomial_kernel(degree: int) -> Kernel:
    def kernel(X, X2):
        return np.power(np.dot(X, X2) + 1, degree)
    return kernel

def create_gaussian_kernel(gamma: float) -> Kernel:
    def kernel(X, X2):
        dist = np.linalg.norm(X - X2, 2)
        return exp(- dist / gamma)
    return kernel

def load_dataset() -> np.ndarray:
    dataset = pd.read_csv('datasets/dataset.csv').values
    np.random.shuffle(dataset)
    return dataset

def plot_feature_correlation(X: np.ndarray):
    _, features = X.shape
    for i in range(features):
        for j in range(i):
            sorted_data = X[X[:, i].argsort()]
            plt.plot(sorted_data[:, i], sorted_data[:, j], 'o')
            plt.xlabel(f'Feature {i}')
            plt.ylabel(f'Feature {j}')
            plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=31415, type=int)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--remove-outliers', action='store_true')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    dataset = load_dataset()
    dataset_size, _ = dataset.shape
    if args.remove_outliers:
        # REPORT: another approach I tried is removing the outliers from the dataset but it is already sufficently cleaned
        #         in fact using the Z-score methods and removing 265 outliers over a dataset size of 10_000_000
        #         affects the performance of the model in a minimum way with no significative changes, in fact in some cases it is (even if only slightly) worsening
        #         comparison data table
        # remove outliers from the dataset using the Z-score method:
        # we calculate the score for each value as Z = (x - u) / o where u is the mean and o is the variance of the data on the feature
        # then we remove all the points with a Z-score greater or equals than 3 in absolute value
        mean = np.mean(dataset, axis=0)
        std = np.std(dataset, axis=0)
        z_score = (dataset - mean) / std
        if args.verbose:
            # DEBUG: print how many outliers we remove
            outliers = dataset[np.any(np.abs(z_score) >= 3, axis=1)]
            n_outliers, _ = outliers.shape
            print(f"preprocessing: removed {n_outliers} outliers on a dataset of {dataset_size} elements")
        dataset = dataset[np.any(np.abs(z_score) < 3, axis=1)]

    # split the dataset in training and test set
    train_set, test_set = split_train_test_set(dataset, training_size=0.8)
    train_set, validation_set = split_train_test_set(train_set, training_size=0.25)
    train_size, _ = train_set.shape
    validation_size, _ = validation_set.shape
    test_size, _ = test_set.shape
    # split datapoints and labels into different arrays for training and test set
    X_train = train_set[:, :-1]
    y_val = validation_set[:, -1]
    X_val = validation_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    # REPORT: NOTE: show the theoretical bound to justify why we use scaling or stadardization (X the radius on the bound for OGD on strongly convex funct)
    # try which one is the best and justify the choice, or maybe better try both and show which algorithm perform better
    # rescale the data to fit in a normal distribution
    X_train, X_val, X_test = standardise(X_train, X_val, X_test)
    # preprocessing: show if some feature are correlated
    
    # REPORT: plotting the feature on the training set on both axis to spot correlation I observed that the feature 2 and 5 have a linear correlation (with a negative coefficent)
    #         as the feature 5 and 9 (with positive coefficent). One possibility in this case during the preprocessing of the data
    #         is to remove the correlated features and leave only one of them to avoid redundancy of the data.
    #         I don't follow this approach because there is a sensibile noise in the correlation and removing some features can lead 
    #         to removing this noise that can encode important information on the model
    # plot_feature_correlation(X_train)
    
    # CLEANUP: add 1 fixed feature to X_train and X_test to express non omogeneous linear separator
    X_train = np.column_stack((X_train, np.ones(train_size)))
    X_val   = np.column_stack((X_val, np.ones(validation_size)))
    X_test  = np.column_stack((X_test, np.ones(test_size)))
    
    perceptron = train_perceptron(X_train, y_train, max_epochs=20)
    print('[Perceptron]')
    print(f"trained perceptron: {perceptron.features}")
    training_error = set_error(zero_one_loss, perceptron, X_train, y_train)
    print(f"training error for perceptron: {training_error}")
    predictions =perceptron.predict(X_test)
    test_error = set_error(zero_one_loss, perceptron, X_test, y_test)
    print(f"test error for perceptron: {test_error}\n")
    
    validation_error = partial(set_error, zero_one_loss, points=X_val, labels=y_val)

    print('[Pegasos]')
    search_result = grid_search(pegasos, X_train, y_train, validation_error, 
                                regularization_coefficent=[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                rounds=(100_000,))
    print(f'best validation error: {search_result.objective}')
    print(f'best hyperparameters: {search_result.parameters}') 
    print(f"grid cv pegasos: {search_result.predictor.features}")
    test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
    print(f"test error for cv pegasos: {test_error}\n")
    
    print('[Regularized Logistic Regression]')
    search_result = grid_search(train_regularized_logistic_classification, 
                                X_train, y_train, validation_error, 
                                regularization_coefficent=[0.1, 1, 10, 100, 1000], 
                                rounds=(100_000,))
    print(f'best validation error: {search_result.objective}')
    print(f'best hyperparameters: {search_result.parameters}') 
    print(f"grid cv logistic regression: {search_result.predictor.features}")
    test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
    print(f"test error for cv logistic regression: {test_error}\n")
    
    # print('[Kernelized Pegasos]')
    # alphas = kernelized_pegasos(X_train, y_train, create_polynomial_kernel(2), regularization_coefficent=0.01, rounds=10_000)
    # print(np.min(alphas))
    # print(np.max(alphas))
    # print(np.mean(alphas))
    # print('\n')

    # try perceptron and svm with polynomial feature expansion of degree 2
    X_train = polynomial_feature_expansion(X_train, 2)
    X_val = polynomial_feature_expansion(X_val, 2)
    X_test = polynomial_feature_expansion(X_test, 2)

    print('[Feature Expanded Perceptron]')
    perceptron = train_perceptron(X_train, y_train, max_epochs=20)
    print(f"trained perceptron with polynomial feature expansion: {perceptron.features}")
    training_error = set_error(zero_one_loss, perceptron, X_train, y_train)
    print(f"training error for perceptron with polynomial feature expansion: {training_error}")
    test_error = set_error(zero_one_loss, perceptron, X_test, y_test)
    print(f"test error for perceptron with polynomial feature expansion: {test_error}\n")
    
    validation_error = partial(set_error, zero_one_loss, points=X_val, labels=y_val)

    print('[Feature Expanded Pegasos]')
    search_result = grid_search(pegasos, X_train, y_train, validation_error, 
                                regularization_coefficent=[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                rounds=(1_000_000,))
    print(f'best validation error: {search_result.objective}')
    print(f'best hyperparameters: {search_result.parameters}') 
    print(f"trained pegasos with cv and polynomial feature expansion: {search_result.predictor.features}")
    training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
    print(f"training error for pegasos with cv and polynomial feature expansion: {training_error}")
    test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
    print(f"test error for pegasos with cv and polynomial feature expansion: {test_error}\n")

    print('[Feature expanded Regularized Logistic Regression]')
    search_result = grid_search(train_regularized_logistic_classification, 
                                X_train, y_train, validation_error, 
                                regularization_coefficent=[0.1, 1, 10, 100, 1000], 
                                rounds=(100_000,))
    print(f'best validation error: {search_result.objective}')
    print(f'best hyperparameters: {search_result.parameters}') 
    print(f"grid cv logistic regression with feature expansion: {search_result.predictor.features}")
    training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
    print(f"training error for cv logistic regression polynomial feature expansion: {training_error}")
    test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
    print(f"test error for cv logistic regression with feature expansion: {test_error}\n")


if __name__ == '__main__':
    main()