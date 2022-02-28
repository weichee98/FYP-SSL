import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


def corr_mx_flatten(X):
    """
    returns upper triangluar matrix of each sample in X

    option 1:
    X.shape == (num_sample, num_feature, num_feature)
    X_flattened.shape == (num_sample, num_feature * (num_feature - 1) / 2)

    option 2:
    X.shape == (num_feature, num_feature)
    X_flattend.shape == (num_feature * (num_feature - 1) / 2,)
    """
    upper_triangular_idx = np.triu_indices(X.shape[1], 1)
    if len(X.shape) == 3:
        X = X[:, upper_triangular_idx[0], upper_triangular_idx[1]]
    else:
        X = X[upper_triangular_idx[0], upper_triangular_idx[1]]
    return X


def make_dataset(X, y, d=None, **kwargs):
    """
    X.shape == (num_samples, num_features)
    y.shape == (num_samples,)
    """
    node_features = torch.tensor(X).type(torch.get_default_dtype())
    graph = Data(x=node_features, y=torch.tensor(y))
    if d is not None:
        le = LabelEncoder()
        d = le.fit_transform(d)
        graph.d = torch.tensor(d)
    for k, v in kwargs.items():
        setattr(graph, k, v)
    return graph
