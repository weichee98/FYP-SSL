import numpy as np
from utils.data import *


def load_GCN_data(
    X,
    Y,
    ages,
    genders,
    ssl,
    labeled_train_indices,
    test_indices,
    sites=None,
    n_ssl=None,
):
    X_flattened = corr_mx_flatten(X)
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        # if SSL is used, all subjects from all sites are used to create graph
        A = get_pop_A(X_flattened, ages, genders)
        data = make_population_graph(X_flattened, A, Y.argmax(axis=1))
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
    else:
        # if SSL is not used, only subjects from largest site is used to create graph
        # adjust the indices accordingly to match the subject used in the graph
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        A = get_pop_A(X_flattened[all_indices], ages[all_indices], genders[all_indices])
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
    X, Y, ssl, labeled_train_indices, test_indices, sites=None, n_ssl=None
):
    X_flattened = corr_mx_flatten(X)
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        data = make_dataset(X_flattened, Y.argmax(axis=1))
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_dataset(X_flattened[all_indices], Y[all_indices].argmax(axis=1))
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_DIVA_data(X, Y, sites, ssl, labeled_train_indices, test_indices, n_ssl=None):
    X_flattened = corr_mx_flatten(X)
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        data = make_dataset(X_flattened, Y.argmax(axis=1), sites)
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_dataset(
            X_flattened[all_indices], Y[all_indices].argmax(axis=1), sites[all_indices]
        )
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_VAESDR_data(
    X, Y, sites, age, gender, ssl, labeled_train_indices, test_indices, n_ssl=None
):
    X_flattened = corr_mx_flatten(X)
    age = np.expand_dims(age, axis=1)
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        data = make_dataset(
            X_flattened,
            Y.argmax(axis=1),
            sites,
            age=torch.tensor(age).type(torch.get_default_dtype()),
            gender=torch.tensor(gender).long(),
        )
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        data = make_dataset(
            X_flattened[all_indices],
            Y[all_indices].argmax(axis=1),
            sites[all_indices],
            age=torch.tensor(age[all_indices]).type(torch.get_default_dtype()),
            gender=torch.tensor(gender[all_indices]).long(),
        )
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    return data, labeled_train_indices, all_train_indices, test_indices


def load_FFN_data(
    X,
    Y,
    ages,
    genders,
    ssl,
    labeled_train_indices,
    test_indices,
    sites=None,
    n_ssl=None,
):
    X_flattened = corr_mx_flatten(X)
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        # if SSL is used, all subjects from all sites are used to create graph
        A = get_pop_A(X_flattened, ages, genders)
        data = make_population_graph(X_flattened, A, Y.argmax(axis=1))
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X_flattened)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
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
    X,
    Y,
    ssl,
    labeled_train_indices,
    test_indices,
    X_ts=None,
    num_process=1,
    verbose=False,
    sites=None,
    n_ssl=None,
):
    if ssl and (n_ssl is None or (isinstance(n_ssl, int) and n_ssl > 0)):
        data = make_graph_dataset(
            X, Y.argmax(axis=1), X_ts, num_process=num_process, verbose=verbose
        )
        if isinstance(ssl, (list, tuple)):
            unlabeled_train_indices = np.argwhere(np.isin(sites, ssl)).flatten()
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
        else:
            all_train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
        if n_ssl is not None:
            unlabeled_train_indices = np.setdiff1d(
                all_train_indices, labeled_train_indices
            )
            unlabeled_train_indices = np.random.choice(
                unlabeled_train_indices, size=n_ssl, replace=False
            )
            all_train_indices = np.union1d(
                unlabeled_train_indices, labeled_train_indices
            )
    else:
        all_indices = np.concatenate([labeled_train_indices, test_indices], axis=0)
        if X_ts is not None:
            X_ts = X_ts[all_indices]
        data = make_graph_dataset(
            X[all_indices],
            Y[all_indices].argmax(axis=1),
            X_ts,
            num_process=num_process,
            verbose=verbose,
        )
        n_train = len(labeled_train_indices)
        n_test = len(test_indices)
        labeled_train_indices = np.array(range(n_train))
        test_indices = np.array(range(n_train, n_train + n_test))
        all_train_indices = None
    data = make_graph_dataloader(
        data, labeled_train_indices, all_train_indices, test_indices
    )
    return data, labeled_train_indices, all_train_indices, test_indices


def load_GNN_data(
    X, Y, train_indices, test_indices, X_ts=None, num_process=1, verbose=False
):
    all_indices = np.concatenate([train_indices, test_indices], axis=0)
    if X_ts is not None:
        X_ts = X_ts[all_indices]
    data = make_graph_dataset(
        X[all_indices],
        Y[all_indices].argmax(axis=1),
        X_ts,
        num_process=num_process,
        verbose=verbose,
    )
    n_train = len(train_indices)
    n_test = len(test_indices)
    train_indices = np.array(range(n_train))
    test_indices = np.array(range(n_train, n_train + n_test))
    data = make_graph_dataloader(data, train_indices, None, test_indices)
    return data, train_indices, None, test_indices
