import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ADHD import load_data_fmri, get_sites
from utils.distribution import SiteDistribution


def sorting(df):
    col_idx = np.argmax(np.var(df.values, axis=1) * np.var(df.values, axis=0))
    idx = np.argsort(df.values[col_idx, :] * df.values[:, col_idx])
    columns = df.columns[idx].tolist()
    return columns


def plot_heatmap(df, metric):
    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    sb.heatmap(
        df, cmap="Blues", square=True, annot=True, 
        fmt=".3f", ax=ax, cbar=False
    )
    ax.set_xlabel("P")
    ax.set_ylabel("Q")
    if metric == SD.METRIC.KL:
        ax.set_title("KL(P||Q)")
    elif metric == SD.METRIC.JS:
        ax.set_title("JS DIVERGENCE")
    elif metric == SD.METRIC.HELLINGER:
        ax.set_title("HELLINGER DISTANCE")
    else:
        raise NotImplementedError("title not set for metric {}".format(metric))
    plt.tight_layout()
    return f


def plot_distribution(X, sites, metric, method, columns=None, **kwargs):
    dist_diff = SD.distribution_heatmap(X, sites, metric, method, **kwargs)
    dist_diff = pd.DataFrame(dist_diff)
    if columns is None:
        columns = sorting(dist_diff)
    dist_diff = dist_diff.loc[columns, :][columns]
    f = plot_heatmap(dist_diff, metric)
    return f


if __name__ == "__main__":

    """
    plot kl divergence heatmap
    """
    X, y = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()
    columns = [
        "NI", "KKI", "PITT", "NYU", "WUSTL", "PKU", "OHSU"
    ]

    for metric in SD.METRIC:
        for method in SD.METHOD:
            try:
                f = plot_distribution(
                    X, sites, metric, method,
                    bins=1000, columns=columns
                )
                f.savefig("distribution_{}_{}.png".format(metric, method))
            except Exception as e:
                print(e)
