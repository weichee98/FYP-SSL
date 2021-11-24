import torch
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from torch_sparse.tensor import SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


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


def make_population_graph(X, A, y, min_weight=0, **kwargs):
    """
    X.shape == (num_samples, num_features)
    y.shape == (num_samples,)
    """
    node_features = torch.tensor(X) # (num_nodes, num_features)
    adj_t = SparseTensor.from_dense(torch.tensor(A), has_value=True)
    d = Data(
        x=node_features, adj_t=adj_t, y=torch.tensor(y)
    )
    for k, v in kwargs.items():
        setattr(d, k, v)
    return d


def make_dataset(X, y, d=None, **kwargs):
    """
    X.shape == (num_samples, num_features)
    y.shape == (num_samples,)
    """
    node_features = torch.tensor(X) # (num_nodes, num_features)
    graph = Data(
        x=node_features, y=torch.tensor(y)
    )
    if d is not None:
        le = LabelEncoder()
        d = le.fit_transform(d)
        graph.d = torch.tensor(d)
    for k, v in kwargs.items():
        setattr(graph, k, v)
    return graph


def make_graph_dataset(X, y, num_process=1, verbose=False):
    """
    X.shape == (num_samples, num_nodes, num_nodes)
    y.shape == (num_samples,)
    """
    def task(x, y):
        node_features = torch.tensor(x)
        adj = SparseTensor.from_dense(node_features, has_value=True)
        d = Data(
            x=node_features, y=torch.tensor([y]),
            adj_t = adj
        )
        return d

    if verbose:
        pbar = tqdm(range(y.shape[0]), desc="Make Graph Dataset")
    else:
        pbar = range(y.shape[0])
    dataset = Parallel(n_jobs=num_process)(
        delayed(task)(X[i], y[i])
        for i in pbar
    )

    """
    dataset: list[torch_geometric.data.Data]
        a list of graphs, each with attributes
        - x: (num_nodes, num_features) - node features
        - y: (1,) - graph label
        - pos: (num_nodes,) - node index
    """
    return dataset


def make_graph_dataloader(data, labeled_idx, all_idx, test_idx):
    labeled_dl = [data[i] for i in labeled_idx]

    if all_idx is None:
        unlabeled_idx = np.array([])
    else:
        unlabeled_idx = np.setdiff1d(all_idx, labeled_idx)
    if unlabeled_idx.shape[0] > 0:
        unlabeled_dl = [data[i] for i in unlabeled_idx]
    else:
        unlabeled_dl = None
    
    test_dl = [data[i] for i in test_idx]
    return labeled_dl, unlabeled_dl, test_dl