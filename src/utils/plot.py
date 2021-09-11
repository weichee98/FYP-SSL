import io
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import squareform


def plot_group_corr_mat(groups, num_process=1, verbose=False):
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

    f, ax = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    cmap = "RdBu"
    vmin, vmax = -1, 1

    def task(i, group):
        idx = ax_idx(i)
        f = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        mat = squareform(groups[group])
        ax.set_title("{} mean".format(group))
        ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        
        buf = io.BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_img = deepcopy(Image.open(buf))
        buf.close()
        plt.close()
        return idx, pil_img

    if verbose:
        pbar = tqdm(enumerate(groups), total=len(groups))
    else:
        pbar = enumerate(groups)
    img_list = Parallel(num_process)(
        delayed(task)(i, group) for i, group in pbar
    )

    if verbose:
        pbar = tqdm(img_list)
    else:
        pbar = img_list
    for idx, img in pbar:
        if idx is None:
            ax.imshow(img)
            ax.axis("off")
        else:
            ax[idx].imshow(img)
            ax[idx].axis("off")

    plt.tight_layout()
    f.colorbar(
        ScalarMappable(Normalize(vmin, vmax, True), cmap=cmap),
        ax=ax, orientation='horizontal', pad=0.05, aspect=80, fraction=0.05
    )
    return f


def plot_group_kde(groups, num_process=1, verbose=False):
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

    def task(r, c):
        g = unique_groups[r]
        f = plt.figure(figsize=(7, 5))
        ax = plt.axes()
        ax.set_xlabel("ROI correlation")
        sb.kdeplot(groups[g], shade=True, ax=ax, label=g)
        if r == c:
            ax.legend([g])
        else:
            g = unique_groups[c]
            sb.kdeplot(groups[g], shade=True, ax=ax, label=g)
            ax.legend()

        buf = io.BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_img = deepcopy(Image.open(buf))
        buf.close()
        plt.close()
        return r, c, pil_img

    if verbose:
        pbar = tqdm(zip(rows, cols), total=len(rows))
    else:
        pbar = zip(rows, cols)
    img_list = Parallel(num_process)(
        delayed(task)(r, c) for r, c in pbar
    )

    if verbose:
        pbar = tqdm(img_list)
    else:
        pbar = img_list
    for r, c, img in pbar:
        ax[r, c].imshow(img)
        ax[r, c].axis("off")
        ax[c, r].axis("off")

    plt.tight_layout()
    return f
