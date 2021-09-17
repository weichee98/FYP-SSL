import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from ABIDE import load_data_fmri, get_sites
from utils.distribution import SiteDistribution


def sorting(df):
    col_idx = np.argmax(np.var(df.values, axis=1) * np.var(df.values, axis=0))
    idx = np.argsort(df.values[col_idx, :] * df.values[:, col_idx])
    columns = df.columns[idx].tolist()
    return columns


def plot_heatmap(df):
    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    sb.heatmap(
        df, cmap="Blues", square=True, annot=True, 
        fmt=".3f", ax=ax, cbar=False
    )
    ax.set_xlabel("P")
    ax.set_ylabel("Q")
    ax.set_title("KL(P||Q)")
    plt.tight_layout()
    return f


if __name__ == "__main__":

    """
    plot kl divergence heatmap
    """
    X, _ = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()

    dist_diff = pd.DataFrame(SD.kl_gauss_distribution_heatmap(X, sites))
    columns = sorting(dist_diff)
    dist_diff = dist_diff.loc[columns, :][columns]
    f = plot_heatmap(dist_diff)
    f.savefig("distribution_kldiv_normal.png")

    bins = [100, 200, 500, 1000]
    for bin in bins:
        dist_diff = pd.DataFrame(SD.kl_hist_distribution_heatmap(X, sites, bin))
        dist_diff = dist_diff.loc[columns, :][columns]
        f = plot_heatmap(dist_diff)
        f.savefig("distribution_kldiv_hist_{}bins.png".format(bin))

    """
    plot kde pairplot
    """
    from utils.plot import plot_group_corr_mat, plot_group_kde
    site_mean = SD.get_site_mean(X, sites, fisher=True)

    # f = plot_group_corr_mat(site_mean, num_process=10, verbose=True)
    # f.savefig("site_mean_corr_mat.png")
    f = plot_group_kde(site_mean, num_process=10, verbose=True, group_order=columns)
    f.savefig("site_distribution.png")

    # X_harmonized, _ = load_data_fmri(True)
    # site_mean = SD.get_site_mean(X_harmonized, sites, fisher=True)

    # f = plot_group_corr_mat(site_mean, num_process=10, verbose=True)
    # f.savefig("combat_site_mean_corr_mat.png")
    # f = plot_group_kde(site_mean, num_process=10, verbose=True, group_order=columns)
    # f.savefig("combat_site_distribution.png")