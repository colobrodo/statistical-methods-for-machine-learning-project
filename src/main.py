from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Any
from math import inf, exp

import argparse
import logging
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernel import Kernel, GaussianKernel, PolynomialKernel
from predictor import Predictor, LinearPredictor
from perceptron import train_kernelized_perceptron, train_perceptron


def split_dataset(dataset: np.ndarray, training_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the dataset into two subset
    
    :param dataset: the whole dataset
    :param training_size: a factor between 0 (only training data) and 1 
    (only test data) that indicates how much data is adibited to the training set percentage"""
    m = dataset.shape[0]
    training_size = int(m * training_size)
    return dataset[:training_size], dataset[training_size:] 

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Given an already splitted dataset into training set (`X_train`), and test set (`X_test`) 
    standardize all the values based on the mean and standard deviation computed
    on the **training set**"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm

def scale(X_train: np.ndarray, X_test: np.ndarray):
    """Given an already splitted dataset into training set (`X_train`), and test set (`X_test`) 
    rescale all the values based on the minimum and maximum computed on the **training set**"""
    min_ = np.min(X_train, axis=0)
    max_ = np.max(X_train, axis=0)
    X_train_scaled = (X_train - min_) / (max_ - min_)
    X_test_scaled = (X_test - min_) / (max_ - min_)
    return X_train_scaled, X_test_scaled

def zero_one_loss(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes the zero-one loss over each pair of the elements in the array `labels` and `predictions`.   
    The array should have the same size or should be broadcastable"""
    return np.not_equal(labels, predictions).astype(int)

def set_error(loss, predictor: Predictor, points: np.ndarray, labels: np.ndarray) -> float:
    # set size
    m = len(labels)
    predictions = predictor(points)
    return np.sum(loss(labels, predictions)) / m

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

def kernelized_pegasos(training_points: np.ndarray, training_labels: np.ndarray, kernel: Kernel, regularization_coefficent=0.1, rounds=1000) -> Predictor:
    samples, _ = training_points.shape
    alpha = np.zeros(samples)
    # NOTE: t the index of current round are 1-based in the for loop to avoid division by zero
    for t in range(1, rounds + 1):
        random_index = random.randint(0, samples - 1)
        # choose the random training point (x_it, y_it)
        x_it = training_points[random_index]
        y_it = training_labels[random_index]
        learning_rate = 1 / (regularization_coefficent * t)
        # TODO: should we exclude somehow the alpha for the current index? 
        if y_it * learning_rate * np.dot(np.multiply(alpha, training_labels), kernel(training_points, x_it)) < 1:
            alpha[random_index] += 1
    # TODO: we should generalize the predictor into a protocol or something like that and create subclass for linear and the two kernel algorithm
    def predict(X: np.ndarray) -> np.ndarray:
        k = kernel(training_points, X)
        d = np.dot(np.multiply(alpha, training_labels), k)
        return np.sign(d)
    return predict

def sigmoid(z: float) -> float:
    """Computes the sigmoid function over a scalar `z`"""
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
    """The result of an hyperparameter search: it contains a resulting predictor function, the objective value
    and a dict that maps each parameter name to the hyperparameter that lead to this result"""
    def __init__(self, configuration: dict[str, Any], predictor: Predictor, objective: float) -> None:
        self.parameters = configuration
        self.predictor = predictor
        self.objective = objective

# TODO: types and doc string, also for single parameter
def grid_search(algorithm, training_points, training_labels, **hyperparameters) -> HyperparameterSearchResult:
    X_dev, X_val = split_dataset(training_points, 0.75)
    y_dev, y_val = split_dataset(training_labels, 0.75)
    best_objective = inf
    best_configuration = None
    # TODO: CLEANUP
    for parameter_values in product(*hyperparameters.values()):
        # NOTE: from python version 3.7 the dictionary are garanteed to preserve insersion order during iteration
        #       for this reason we can iterate this using zip
        hyperparameters_configuration = dict(zip(hyperparameters, parameter_values))
        predictor = algorithm(X_dev, y_dev, **hyperparameters_configuration)
        objective = set_error(zero_one_loss, predictor, X_val, y_val)
        logging.debug(f"{hyperparameters_configuration} -> {objective}")
        if objective < best_objective:
            best_objective = objective
            best_configuration = hyperparameters_configuration

    logging.debug("terminated hyperparameter search\n")
    # we should retrain the algorithm on the whole training set 
    predictor = algorithm(training_points, training_labels, **best_configuration)
    return HyperparameterSearchResult(best_configuration, predictor, best_objective)

def polynomial_feature_expansion(X: np.ndarray, n: int) -> np.ndarray:
    """Computes the polynomial feature expansion for the array `X` of degree `n`"""
    samples, features = X.shape
    # Generate combinations of feature indices up to the given degree
    combinations = []
    for d in range(1, n + 1):
        combinations.extend(combinations_with_replacement(range(features), d))
    # TODO:
    # Create a list to hold the new features
    poly_features = np.empty((samples, len(combinations)), dtype=X.dtype)
    # Generate the new features
    for i, comb in enumerate(combinations):
        poly_features[:, i] = np.prod(X[:, comb], axis=1)
    return poly_features


def load_dataset(path: str) -> np.ndarray:
    """Load the csv dataset located at `path` in a randomized order"""
    dataset = pd.read_csv(path).values
    np.random.shuffle(dataset)
    return dataset

def plot_feature_correlation(X: np.ndarray):
    """Plot all the combination of features using the first on the x axis and the second as y.   
    This is usefull to spot visually eventually correlation of data between different feature.
    """
    _, features = X.shape
    for i in range(features):
        for j in range(i):
            sorted_data = X[X[:, i].argsort()]
            plt.plot(sorted_data[:, i], sorted_data[:, j], 'o')
            plt.xlabel(f'Feature {i}')
            plt.ylabel(f'Feature {j}')
            plt.show()

def preprocessing(dataset: np.ndarray, args):
    dataset_size, _ = dataset.shape
    # REPORT: also say that I check for duplicates and don't find any of them
    if args.remove_outliers:
        # REPORT: another approach I tried is removing the outliers from the dataset but it is already sufficently cleaned
        #         in fact using the Z-score methods and removing 265 outliers over a dataset size of 10_000
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
            logging.debug(f"preprocessing: removed {n_outliers} outliers on a dataset of {dataset_size} elements")
        dataset = dataset[np.any(np.abs(z_score) < 3, axis=1)]

    # split the dataset in training and test set
    train_set, test_set = split_dataset(dataset, training_size=0.8)
    train_size, _ = train_set.shape
    test_size, _ = test_set.shape
    # split datapoints and labels into different arrays for training and test set
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    # REPORT: NOTE: show the theoretical bound to justify why we use scaling or stadardization (X the radius on the bound for OGD on strongly convex funct)
    # try which one is the best and justify the choice, or maybe better try both and show which algorithm perform better
    # rescale the data to fit in a normal distribution
    if args.preprocess == 'standardize':
        logging.debug('preprocessing: feature rescaling standardization')
        X_train, X_test = standardize(X_train, X_test)
    elif args.preprocess == 'normalize':
        logging.debug('preprocessing: feature rescaling with normalization')
        X_train, X_test = scale(X_train, X_test)
    else:
        logging.debug('preprocessing: no feature rescaling')
    
    # preprocessing: feature augmentation
    # add 1 fixed feature to X_train and X_test to express non omogeneous linear separator
    X_train = np.column_stack((X_train, np.ones(train_size)))
    X_test  = np.column_stack((X_test, np.ones(test_size)))

    training_set = X_train, y_train
    test_set = X_test, y_test
    return training_set, test_set

def train_predictor(dataset: np.ndarray, args):
    """Train a predictor on the dataset based on the given arguments and 
    serialize the result to the output argument"""
    training_set, test_set = preprocessing(dataset, args)
    X_train, y_train = training_set
    X_test, y_test = test_set
    
    if args.algorithm == 'perceptron':
        predictor = train_perceptron(X_train, y_train, max_epochs=20)
        logging.info('[Perceptron]')
        logging.info(f"trained perceptron: {predictor.features}")
        training_error = set_error(zero_one_loss, predictor, X_train, y_train)
        logging.debug(f"training error for perceptron: {training_error}")
        test_error = set_error(zero_one_loss, predictor, X_test, y_test)
        logging.info(f"test error for perceptron: {test_error}\n")
    elif args.algorithm == 'pegasos':
        logging.info('[Pegasos]')
        search_result = grid_search(pegasos, X_train, y_train, 
                                    regularization_coefficent=[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                    rounds=(100_000,))
        logging.debug(f'best validation error: {search_result.objective}')
        logging.debug(f'best hyperparameters: {search_result.parameters}') 
        logging.info(f"grid cv pegasos: {search_result.predictor.features}")
        test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
        logging.info(f"test error for cv pegasos: {test_error}\n")
        predictor = search_result.predictor
    elif args.algorithm == 'logistic-regression':
        logging.info('[Regularized Logistic Regression]')
        search_result = grid_search(train_regularized_logistic_classification, 
                                    X_train, y_train, 
                                    regularization_coefficent=[0.1, 1, 10, 100, 1000], 
                                    rounds=(100_000,))
        logging.debug(f'best validation error: {search_result.objective}')
        logging.debug(f'best hyperparameters: {search_result.parameters}') 
        logging.info(f"grid cv logistic regression: {search_result.predictor.features}")
        test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
        logging.info(f"test error for cv logistic regression: {test_error}\n")
        predictor = search_result.predictor
    elif args.algorithm == 'kernelized-perceptron':
        logging.info('[Kernelized Perceptron]')
        kernels = [PolynomialKernel(degree) for degree in range(1, 5)]
        kernels += [GaussianKernel(gamma) for gamma in (0.01, 0.1, 1, 10)]
        search_result = grid_search(train_kernelized_perceptron, 
                                    X_train, y_train, 
                                    kernel=kernels)
        logging.info(f'best hyperparameters: {search_result.parameters}') 
        logging.debug(f'validation error for kernelized perceptron: {search_result.objective}')
        training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
        logging.debug(f'training error for kernelized perceptron: {training_error}')
        test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
        logging.info(f'test error for kernelized perceptron: {test_error}\n')
        predictor = search_result.predictor
    elif args.algorithm == 'kernelized-pegasos': 
        logging.info('[Kernelized Pegasos]')
        kernels = [PolynomialKernel(degree) for degree in range(1, 5)]
        kernels += [GaussianKernel(gamma) for gamma in (0.01, 0.1, 1, 10)]
        search_result = grid_search(kernelized_pegasos, 
                                    X_train, y_train,
                                    regularization_coefficent=[0.001, 0.01, 0.1, 1, 10, 100], 
                                    kernel=kernels,
                                    rounds=(100_000,))
        logging.debug(f'validation error for kernelized pegasos: {search_result.objective}')
        logging.debug(f'best hyperparameters: {search_result.parameters}') 
        training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
        logging.debug(f'training error for kernelized pegasos: {training_error}')
        test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
        logging.info(f'test error for kernelized pegasos: {test_error}\n')
        predictor = search_result.predictor
    else:
        # try perceptron and svm with polynomial feature expansion of degree 2
        X_train = polynomial_feature_expansion(X_train, 2)
        X_test = polynomial_feature_expansion(X_test, 2)
        if args.algorithm == 'feature-expanded-perceptron':
            logging.info('[Feature Expanded Perceptron]')
            perceptron = train_perceptron(X_train, y_train, max_epochs=20)
            logging.info(f"trained perceptron with polynomial feature expansion: {perceptron.features}")
            training_error = set_error(zero_one_loss, perceptron, X_train, y_train)
            logging.debug(f"training error for perceptron with polynomial feature expansion: {training_error}")
            test_error = set_error(zero_one_loss, perceptron, X_test, y_test)
            logging.info(f"test error for perceptron with polynomial feature expansion: {test_error}\n")
        elif args.algorithm == 'feature-expanded-pegasos':
            logging.info('[Feature Expanded Pegasos]')
            search_result = grid_search(pegasos, X_train, y_train,
                                        regularization_coefficent=[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                        rounds=(1_000_000,))
            logging.debug(f'best validation error: {search_result.objective}')
            logging.debug(f'best hyperparameters: {search_result.parameters}') 
            logging.info(f"trained pegasos with cv and polynomial feature expansion: {search_result.predictor.features}")
            training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
            logging.debug(f"training error for pegasos with cv and polynomial feature expansion: {training_error}")
            test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
            logging.info(f"test error for pegasos with cv and polynomial feature expansion: {test_error}\n")
        elif args.algorithm == 'feature-expanded-logistic-regression':
            logging.info('[Feature expanded Regularized Logistic Regression]')
            search_result = grid_search(train_regularized_logistic_classification, 
                                        X_train, y_train,
                                        regularization_coefficent=[0.1, 1, 10, 100, 1000], 
                                        rounds=(100_000,))
            logging.debug(f'best validation error: {search_result.objective}')
            logging.debug(f'best hyperparameters: {search_result.parameters}') 
            logging.info(f"grid cv logistic regression with feature expansion: {search_result.predictor.features}")
            training_error = set_error(zero_one_loss, search_result.predictor, X_train, y_train)
            logging.debug(f"training error for cv logistic regression polynomial feature expansion: {training_error}")
            test_error = set_error(zero_one_loss, search_result.predictor, X_test, y_test)
            logging.info(f"test error for cv logistic regression with feature expansion: {test_error}\n")
        else:
            raise NotImplementedError(f'Not implemented algorithm {args.algorithm}')    
    
    with open(args.output, 'wb+') as f:
        pickle.dump(predictor, f)

def run_predictor(dataset: np.ndarray, args):
    """Load a serialized predictor and compute the training and test error.   
    When you run this command you should be aware that you should provide 
    the same seed and the same preprocessing options that you give when you 
    have trained the predictor to have consistent results"""
    training_set, test_set = preprocessing(dataset, args)
    X_train, y_train = training_set
    X_test, y_test = test_set
    with open(args.predictor, 'rb') as f:
        predictor = pickle.load(f)
    training_error = set_error(zero_one_loss, predictor, X_train, y_train)
    logging.info(f"training error for predictor saved at {args.predictor!r}: {training_error}")
    test_error = set_error(zero_one_loss, predictor, X_test, y_test)
    logging.info(f"test error for predictor saved at {args.predictor!r} : {test_error}\n")


def main():
    parser = argparse.ArgumentParser()
    # TODO: add subcommands train and run
    parser.add_argument('-i', '--input', default='./datasets/dataset.csv', type=str, 
                        help="The path for the input dataset")
    parser.add_argument('-s', '--seed', default=31415, type=int,
                        help="The PRNG seed to allow reproducible results")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--remove-outliers', action='store_true',
                        help="If specified remove all the outliers from the "
                        "dataset using the Z-score method")
    parser.add_argument('--preprocess', 
                        choices=('normalize', 'standardize', 'none'),
                        default='standardize')
    
    subparsers = parser.add_subparsers(required=True)
    available_algorithms = [
        'perceptron', 
        'pegasos', 
        'logistic-regression', 
        'feature-expanded-perceptron', 
        'feature-expanded-pegasos', 
        'feature-expanded-logistic-regression', 
        'kernelized-pegasos', 
        'kernelized-perceptron',
        ]
    # define option parameters to train the algorithms
    # TODO: add help for this subcommand
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('algorithm', type=str, 
                            choices=available_algorithms)
    parser_train.add_argument('output', type=str)
    parser_train.set_defaults(func=train_predictor)

    # TODO: add help for this subcommand
    parser_run = subparsers.add_parser('run')
    parser_run.add_argument('predictor', type=str)
    parser_run.set_defaults(func=run_predictor)

    # parse command line arguments
    args = parser.parse_args()

    # setting up logging, choose to show debug message only if verbose is active
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(message)s', level=level)
    # loading and shuffling the dataset    
    np.random.seed(args.seed)
    dataset = load_dataset(args.input)

    # execute routine based on the cli arguments
    args.func(dataset, args)
    


if __name__ == '__main__':
    main()