from __future__ import annotations

import argparse
import logging
import pickle
from itertools import combinations_with_replacement, product
from math import inf
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kernel import GaussianKernel, PolynomialKernel
from pegasos import (kernelized_pegasos, pegasos,
                     train_regularized_logistic_classification)
from perceptron import train_kernelized_perceptron, train_perceptron
from predictor import Predictor
from preprocessing import preprocessing, split_dataset, ScalingType


def zero_one_loss(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Computes the zero-one loss over each pair of the elements in the array `labels` and `predictions`.   
    The array should have the same size or should be broadcastable"""
    return np.not_equal(labels, predictions).astype(int)

def set_error(loss, predictor: Predictor, points: np.ndarray, labels: np.ndarray) -> float:
    # set size
    m = len(labels)
    predictions = predictor(points)
    return np.sum(loss(labels, predictions)) / m

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


def train_predictor(dataset: np.ndarray, args):
    """Train a predictor on the dataset based on the given arguments and 
    serialize the result to the output argument"""
    scaling = {
        'standardize': ScalingType.STANDARDIZE,
        'normalize': ScalingType.NORMALIZE,
        'none': ScalingType.NONE,
    }[args.preprocess]
    training_set, test_set = preprocessing(dataset, scaling, args.remove_outliers)
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