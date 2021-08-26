import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform


def corr_mat_flatten(X):
    upper_triangular_idx = np.triu_indices(X.shape[1], 1)
    if len(X.shape) == 3:
        X = X[:, upper_triangular_idx[0], upper_triangular_idx[1]]
    else:
        X = X[upper_triangular_idx[0], upper_triangular_idx[1]]
    return X


def get_group_distribution(X, groups, fisher=False):
    X = corr_mat_flatten(X)
    if fisher:
        X = np.arctanh(X)
    group_dic = {}
    for group in np.unique(groups):
        idx = np.argwhere(groups == group).flatten()
        group_mean = np.mean(X[idx], axis=0)
        if fisher:
            group_mean = np.tanh(group_mean)
        group_dic[group] = group_mean
    return group_dic


def plot_group_corr_mat(groups):
    num_groups = len(groups)
    num_cols = np.ceil(np.sqrt(num_groups))
    num_rows = np.ceil(num_groups / num_cols)
    num_rows, num_cols = int(num_rows), int(num_cols)

    def ax_idx(i):
        c = i % num_cols
        r = i // num_cols
        if num_rows == 1 and num_cols == 1:
            return None
        if num_rows == 1:
            return c
        if num_cols == 1:
            return r
        return r, c

    f, ax = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * (num_rows + 1)))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    for i, group in enumerate(groups):
        idx = ax_idx(i)
        mat = squareform(groups[group])
        if idx is not None:
            ax[idx].set_title("{} mean".format(group))
            im = ax[idx].imshow(mat, vmin=-1, vmax=1, cmap='RdBu')
        else:
            ax.set_title("{} mean".format(group))
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu')

    f.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=80)
    return f


def plot_group_kde(groups):
    n = len(groups)
    if n == 1:
        g = list(groups)[0]
        f, ax = plt.subplots(1, 1, figsize=(7, 5))
        sb.kdeplot(groups[g], shade=True)
        ax.set_xlabel("ROI correlation")
        ax.set_title(g)
        return f

    unique_groups = list(groups)
    rows, cols = np.triu_indices(n)
    f, ax = plt.subplots(n, n, figsize=(7 * n, 5 * n))

    for r, c in zip(rows, cols):
        g = unique_groups[r]
        ax[r, c].set_xlabel("ROI correlation")
        sb.kdeplot(groups[g], shade=True, ax=ax[r, c], label=g)
        if r == c:
            ax[r, c].legend([g])
            continue
        g = unique_groups[c]
        sb.kdeplot(groups[g], shade=True, ax=ax[r, c], label=g)
        ax[r, c].legend()
        ax[c, r].axis("off")

    return f
