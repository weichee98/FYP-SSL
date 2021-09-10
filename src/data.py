import numpy as np
from utils import *


def load_GCN_data(
        X, Y, ages, genders, ssl, labeled_train_indices, test_indices
    ):
    X_flattened = corr_mx_flatten(X)
    if ssl:
        # if SSL is used, all subjects from all sites are used to create graph
        A = get_pop_A(X_flattened, ages, genders)
        data = make_population_graph(X_flattened, A, Y.argmax(axis=1))
        all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
    else:
        # if SSL is not used, only subjects from largest site is used to create graph
        # adjust the indices accordingly to match the subject used in the graph
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        A = get_pop_A(
            X_flattened[all_indices], ages[all_indices], genders[all_indices]
        )
        data = make_population_graph(
            X_flattened[all_indices], A, Y[all_indices].argmax(axis=1)
        )
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_AE_data(
        X, Y, ssl, labeled_train_indices, test_indices
    ):
    X_flattened = corr_mx_flatten(X)
    if ssl:
        data = make_dataset(X_flattened, Y.argmax(axis=1))
        all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_dataset(X_flattened[all_indices], Y[all_indices].argmax(axis=1))
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_FFN_data(
        X, Y, ages, genders, ssl, labeled_train_indices, test_indices
    ):
    X_flattened = corr_mx_flatten(X)
    if ssl:
        # if SSL is used, all subjects from all sites are used to create graph
        A = get_pop_A(X_flattened, ages, genders)
        data = make_population_graph(X_flattened, A, Y.argmax(axis=1))
        all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
    else:
        # if SSL is not used, only subjects from largest site is used to create graph
        # adjust the indices accordingly to match the subject used in the graph
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_dataset(X_flattened[all_indices], Y[all_indices].argmax(axis=1))
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_GAE_data(
        X, Y, ssl, labeled_train_indices, test_indices, 
        num_process=1, batch_size=None, verbose=False
    ):
    if ssl:
        data = make_graph_dataset(X, Y.argmax(axis=1), num_process, verbose)
        all_train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_graph_dataset(
            X[all_indices], Y[all_indices].argmax(axis=1), num_process, verbose
        )
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    data = make_graph_dataloader(
        data, labeled_train_indices, all_train_indices, test_indices, batch_size
    )
    return data, labeled_train_indices, all_train_indices, test_indices


def load_GNN_data(
        X, Y, train_indices, test_indices, 
        num_process=1, batch_size=None, verbose=False
    ):
    all_indices = np.concatenate([train_indices, test_indices], axis=0)
    data = make_graph_dataset(
        X[all_indices], Y[all_indices].argmax(axis=1), num_process, verbose
    )
    n_train = len(train_indices)
    n_test = len(test_indices)
    train_indices = np.array(range(n_train))
    test_indices = np.array(range(n_train, n_train + n_test))
    data = make_graph_dataloader(
        data, train_indices, None, test_indices, batch_size
    )
    return data, train_indices, None, test_indices