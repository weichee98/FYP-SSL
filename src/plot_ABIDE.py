import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from ABIDE import load_data_fmri, get_sites
from utils.data import get_group_distribution


def kl(p, q):
    eps = min(np.min(q[q > 0]), np.min(p[p > 0]), 1e-5)
    p, q = np.maximum(p, eps), np.maximum(q, eps)
    kl_div = p * np.log(p / q)
    return np.sum(kl_div, axis=0)


def kl_gauss(p_mu, p_var, q_mu, q_var):
    kl = 0.5 * (
        np.log(q_var) - np.log(p_var) + \
        (p_var + (p_mu - q_mu) ** 2) / q_var - 1
    )
    return kl


def smoothed_hist_kldiv(a, b, nbins=200, sigma=1):
    ahist = np.histogram(a, bins=nbins, range=(-1, 1), density=True)[0]
    bhist = np.histogram(b, bins=nbins, range=(-1, 1), density=True)[0]
    asmooth = gaussian_filter(ahist, sigma)
    bsmooth = gaussian_filter(bhist, sigma)
    asmooth = asmooth / np.sum(asmooth)
    bsmooth = bsmooth / np.sum(bsmooth)
    assert np.isclose(asmooth, 1)
    assert np.isclose(bsmooth, 1)
    return kl(asmooth, bsmooth)


def hist_kldiv(a, b, nbins=200):
    ahist = np.histogram(a, bins=nbins, range=(-1, 1), density=True)[0]
    bhist = np.histogram(b, bins=nbins, range=(-1, 1), density=True)[0]
    ahist = ahist / np.sum(ahist)
    bhist = bhist / np.sum(bhist)
    assert np.isclose(np.sum(ahist), 1)
    assert np.isclose(np.sum(bhist), 1)
    return kl(ahist, bhist)


def normal_kldiv(p, q):
    p_mu, p_var = np.mean(p), np.var(p)
    q_mu, q_var = np.mean(q), np.var(q)
    return kl_gauss(p_mu, p_var, q_mu, q_var)


def sorting(df):
    col_idx = np.argmax(np.var(df.values, axis=1) * np.var(df.values, axis=0))
    idx = np.argsort(df.values[col_idx, :] * df.values[:, col_idx])
    columns = df.columns[idx].tolist()
    df = df.loc[columns, :][columns]
    return df


def plot_heatmap(df):
    df = sorting(df)
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
    site_mean = get_group_distribution(
        X, sites, fisher=True
    )
    unique_sites = np.unique(sites)

    bins = [100, 200, 500, 1000]
    for bin in bins:
        dist_diff = dict()
        for s1 in unique_sites:
            dist_diff[s1] = dict()
            for s2 in unique_sites:
                dist_diff[s1][s2] = hist_kldiv(
                    site_mean[s1], site_mean[s2], bin
                )
        dist_diff = pd.DataFrame(dist_diff)
        f = plot_heatmap(dist_diff)
        f.savefig("distribution_kldiv_hist{}.png".format(bin))

    dist_diff = dict()
    for s1 in unique_sites:
        dist_diff[s1] = dict()
        for s2 in unique_sites:
            dist_diff[s1][s2] = normal_kldiv(
                site_mean[s1], site_mean[s2]
            )
    dist_diff = pd.DataFrame(dist_diff)
    f = plot_heatmap(dist_diff)
    f.savefig("distribution_kldiv_normal.png")

    """
    plot kde pairplot
    """
    # from utils.plot import plot_group_corr_mat, plot_group_kde

    # f = plot_group_corr_mat(site_mean, num_process=10, verbose=True)
    # f.savefig("site_mean_corr_mat.png")
    # f = plot_group_kde(site_mean, num_process=10, verbose=True)
    # f.savefig("site_distribution.png")

    # X_harmonized, _ = load_data_fmri(True)
    # site_mean = get_group_distribution(
    #     X_harmonized, sites,
    #     fisher=True
    # )

    # f = plot_group_corr_mat(site_mean, num_process=10, verbose=True)
    # f.savefig("combat_site_mean_corr_mat.png")
    # f = plot_group_kde(site_mean, num_process=10, verbose=True)
    # f.savefig("combat_site_distribution.png")