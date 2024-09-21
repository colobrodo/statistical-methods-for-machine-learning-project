import logging
from enum import Enum, auto

import numpy as np


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

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """Given an already splitted dataset into training set (`X_train`), and test set (`X_test`) 
    rescale all the values based on the minimum and maximum computed on the **training set**"""
    min_ = np.min(X_train, axis=0)
    max_ = np.max(X_train, axis=0)
    X_train_scaled = (X_train - min_) / (max_ - min_)
    X_test_scaled = (X_test - min_) / (max_ - min_)
    return X_train_scaled, X_test_scaled

class ScalingType(Enum):
    STANDARDIZE = auto()
    NORMALIZE = auto()
    NONE = auto()


def preprocessing(dataset: np.ndarray, scaling=ScalingType.STANDARDIZE, remove_outliers=False):
    dataset_size, _ = dataset.shape
    # REPORT: also say that I check for duplicates and don't find any of them
    if remove_outliers:
        # REPORT: another approach I tried is removing the outliers from the dataset but it is already sufficently cleaned
        #         in fact using the Z-score methods and removing 265 outliers over a dataset size of 10000
        #         affects the performance of the model in a minimum way with no significative changes, in fact in some cases it is (even if only slightly) worsening
        #         comparison data table
        # remove outliers from the dataset using the Z-score method:
        # we calculate the score for each value as Z = (x - u) / o where u is the mean and o is the variance of the data on the feature
        # then we remove all the points with a Z-score greater or equals than 3 in absolute value
        mean = np.mean(dataset, axis=0)
        std = np.std(dataset, axis=0)
        z_score = (dataset - mean) / std
        dataset = dataset[np.any(np.abs(z_score) < 3, axis=1)]
        n_no_outliers, _ = dataset.shape
        n_outliers = dataset_size - n_no_outliers
        logging.debug(f"preprocessing: removed {n_outliers} outliers on a dataset of {dataset_size} elements")

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
    if scaling == ScalingType.STANDARDIZE:
        logging.debug('preprocessing: feature rescaling standardization')
        X_train, X_test = standardize(X_train, X_test)
    elif scaling == ScalingType.NORMALIZE:
        logging.debug('preprocessing: feature rescaling with normalization')
        X_train, X_test = normalize(X_train, X_test)
    else:
        assert scaling == ScalingType.NONE
        logging.debug('preprocessing: no feature rescaling')
    
    # preprocessing: feature augmentation
    # add 1 fixed feature to X_train and X_test to express non omogeneous linear separator
    X_train = np.column_stack((X_train, np.ones(train_size)))
    X_test  = np.column_stack((X_test, np.ones(test_size)))

    training_set = X_train, y_train
    test_set = X_test, y_test
    return training_set, test_set