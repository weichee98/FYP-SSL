import os
import traceback
import torch
import torch_geometric
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def on_error(value, print_error_stack=True):
    """
    returns a wrapper which catches error within a function 
    and returns a default value on error
    value: the default value to be returned when error occured
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                if print_error_stack:
                    traceback.print_exc()
                return value
        return wrapper
    return decorator


def corr_mx_flatten(X):
    """
    returns upper triangluar matrix of each sample in X
    X.shape == (num_sample, num_feature, num_feature)
    X_flattened.shape == (num_sample, num_feature * (num_feature - 1) / 2)
    """
    upper_triangular_idx = np.triu_indices(X.shape[1], 1)
    X_flattened = X[:, upper_triangular_idx[0], upper_triangular_idx[1]]
    return X_flattened


def get_pop_A(X, ages, genders):
    """
    X.shape == (num_sample, num_features)
    ages.shape == (num_sample,)
    genders.shape == (num_sample,)
    returns the weighted adj matrix (num_sample, num_sample)
    """
    num_samples = X.shape[0]
    assert ages.shape[0] == num_samples
    assert genders.shape[0] == num_samples

    age_sim = distance.pdist(np.expand_dims(ages, 1))
    age_sim = np.where(age_sim < 2, 1, 0)
    age_sim = distance.squareform(age_sim)

    le = LabelEncoder()
    genders = le.fit_transform(genders)
    gender_sim = distance.pdist(np.expand_dims(genders, 1))
    gender_sim = np.where(gender_sim == 0, 1, 0)
    gender_sim = distance.squareform(gender_sim)

    dist = distance.pdist(X, metric='correlation') 
    dist = distance.squareform(dist)  
    sigma = np.mean(dist)
    dist = np.exp(- dist ** 2 / (2 * sigma ** 2))
    A = dist * (gender_sim + age_sim)
    np.fill_diagonal(A, 0)
    return A


def distance_to_similarity(adj, edge_thres=None):
    adj = (adj - np.mean(adj)) / np.std(adj)
    adj = adj - adj.min()
    adj = np.exp(-(adj ** 2))
    if edge_thres is not None:
        adj [adj < edge_thres] = 0
    return adj


def make_graph(X, A, y, min_weight=0, **kwargs):
    """
    X.shape == (num_samples, num_features)
    A.shape == (num_samples, num_samples); the weighted adj matrix
    y.shape == (num_samples,)
    """
    node_features = torch.tensor(X).float() # (num_nodes, num_features)
    edge_index = torch.tensor(np.argwhere((A > min_weight)).T) # (2, num_edges)
    weights = torch.tensor(A[np.where(A > min_weight)]).float()
    d = torch_geometric.data.Data(
        x=node_features, edge_index=edge_index, 
        edge_attr=weights, y=torch.tensor(y)
    )
    for k, v in kwargs.items():
        setattr(d, k, v)
    return d
