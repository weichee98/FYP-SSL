import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ABIDE import get_labelling_standards, load_data_fmri, get_sites
from utils.distribution import SiteDistribution


def sorting(df):
    col_idx = np.argmax(np.var(df.values, axis=1) * np.var(df.values, axis=0))
    idx = np.argsort(df.values[col_idx, :] * df.values[:, col_idx])
    columns = df.columns[idx].tolist()
    return columns


def plot_heatmap(df, metric, figsize=(15, 15)):
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sb.heatmap(df, cmap="Blues", square=True, annot=True, fmt=".3f", ax=ax, cbar=False)
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


def plot_distribution(
    X, sites, metric, method, columns=None, figsize=(15, 15), **kwargs
):
    dist_diff = SD.distribution_heatmap(X, sites, metric, method, **kwargs)
    dist_diff = pd.DataFrame(dist_diff)
    if columns is None:
        columns = sorting(dist_diff)
    dist_diff = dist_diff.loc[columns, :][columns]
    f = plot_heatmap(dist_diff, metric, figsize)
    return f


def get_groups(X, sites):
    groups = get_labelling_standards()
    groups.pop("CMU")

    group_ids = list(groups.values())
    max_group = max(set(group_ids), key=group_ids.count)
    valid_sites = [site for site in groups if groups[site] == max_group]

    valid_idx = np.isin(sites, valid_sites)
    X = X[valid_idx]
    sites = sites[valid_idx]
    groups = SD.get_site_grouping(X, sites, SD.METRIC.HELLINGER, SD.METHOD.KDE, 0.063)

    group_sites = defaultdict(list)
    for site, group_id in groups.items():
        group_sites[group_id].append(site)
    for group_id, g_sites in group_sites.items():
        group_sites[group_id] = ", ".join(sorted(g_sites))

    func = lambda x: group_sites[groups[x]]
    group_order = sorted(group_sites.values())
    subject_groups = np.vectorize(func)(sites)
    return X, subject_groups, group_order


if __name__ == "__main__":

    """
    plot kl divergence heatmap
    """
    X, y = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()

    columns = [
        "CMU",
        "OHSU",
        "YALE",
        "LEUVEN_1",
        "LEUVEN_2",
        "UCLA_2",
        "UCLA_1",
        "SBL",
        "TRINITY",
        "SDSU",
        "OLIN",
        "PITT",
        "KKI",
        "STANFORD",
        "CALTECH",
        "UM_1",
        "UM_2",
        "MAX_MUN",
        "NYU",
        "USM",
    ]

    # for metric in SD.METRIC:
    #     for method in SD.METHOD:
    #         try:
    #             f = plot_distribution(
    #                 X, sites, metric, method, bins=1000, columns=columns
    #             )
    #             f.savefig("distribution_{}_{}.png".format(metric, method))
    #         except Exception as e:
    #             print(e)

    X, subject_groups, group_order = get_groups(X, sites)
    f = plot_distribution(
        X,
        subject_groups,
        SD.METRIC.HELLINGER,
        SD.METHOD.KDE,
        bins=1000,
        figsize=(10, 10),
        columns=group_order,
    )
    f.savefig("distribution_group_heatmap.png")
