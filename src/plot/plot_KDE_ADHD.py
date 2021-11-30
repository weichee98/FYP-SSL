import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ADHD import load_data_fmri, get_sites, get_labelling_standards
from utils.distribution import SiteDistribution
from utils.plot import plot_group_kde
from collections import defaultdict


SD = SiteDistribution()


def plot_pairwise_site(X, y, sites, columns=None, name="site_distribution"):
    site_mean = SD.get_site_mean(X, sites, fisher=True)
    f = plot_group_kde(site_mean, num_process=10, verbose=True, group_order=columns)
    f.savefig("{}.png".format(name))

    is_diseased = y[:, 1] == 1
    site_mean_control = SD.get_site_mean(X[~is_diseased], sites[~is_diseased], fisher=True)
    site_mean_diseased = SD.get_site_mean(X[is_diseased], sites[is_diseased], fisher=True)
    
    site_mean = dict()
    for s in np.unique(sites):
        site_mean[s] = dict()
        if s in site_mean_control:
            site_mean[s]["control"] = site_mean_control[s]
        if s in site_mean_diseased:
            site_mean[s]["diseased"] = site_mean_diseased[s]

    f = plot_group_kde(site_mean, num_process=10, verbose=True, group_order=columns)
    f.savefig("{}_with_classes.png".format(name))


def plot_pairwise_group(X, y, sites, group_ids, name="group_distribution"):
    """
    group_ids: dict
        key -> site name
        value -> group number
    """
    valid_idx = np.isin(sites, list(group_ids))
    X = X[valid_idx]
    y = y[valid_idx]
    sites = sites[valid_idx]

    """
    group_names: dict
        key -> group number
        value -> list of site names
    """
    group_names = defaultdict(list)
    for site, group in group_ids.items():
        group_names[group].append(site)
    print(group_names)

    group_legend = "\n".join([
        "Group {}:\n{}\n".format(k, group_names[k])
        for k in sorted(group_names)
    ])
    
    def get_group_name(site):
        return "Group {}".format(group_ids[site])
    groups = np.vectorize(get_group_name)(sites)
    group_order = ["Group {}".format(g) for g in sorted(group_names)]
    group_mean = SD.get_site_mean(X, groups, fisher=True)
    f = plot_group_kde(
        group_mean, num_process=10, verbose=True, 
        group_order=group_order
    )
    f.text(0.1, 0.1, group_legend)
    f.savefig("{}.png".format(name))

    is_diseased = y[:, 1] == 1
    group_mean_control = SD.get_site_mean(X[~is_diseased], groups[~is_diseased], fisher=True)
    group_mean_diseased = SD.get_site_mean(X[is_diseased], groups[is_diseased], fisher=True)
    group_mean = dict()
    for s in np.unique(groups):
        group_mean[s] = dict()
        if s in group_mean_control:
            group_mean[s]["control"] = group_mean_control[s]
        if s in group_mean_diseased:
            group_mean[s]["diseased"] = group_mean_diseased[s]
    f = plot_group_kde(
        group_mean, num_process=10, verbose=True, group_order=group_order
    )
    f.text(0.1, 0.1, group_legend)
    f.savefig("{}_with_classes.png".format(name))


def get_groups(X, sites):
    groups = get_labelling_standards()
    groups.pop("CMU")

    group_ids = list(groups.values())
    max_group = max(set(group_ids), key=group_ids.count)
    valid_sites = [site for site in groups if groups[site] == max_group]

    valid_idx = np.isin(sites, valid_sites)
    groups = SD.get_site_grouping(X[valid_idx], sites[valid_idx], SD.METRIC.HELLINGER, SD.METHOD.KDE, 0.063)
    return groups


if __name__ == "__main__":

    """
    plot kl divergence heatmap
    """
    X, y = load_data_fmri()
    X_harmonized, _ = load_data_fmri(harmonized=True)
    sites = get_sites()
    columns = [
        "NI", "KKI", "PITT", "NYU", "WUSTL", "PKU", "OHSU"
    ]

    plot_pairwise_site(X, y, sites, columns, "site_distribution")
    plot_pairwise_site(X_harmonized, y, sites, columns, "site_distribution_combat")

    # group_ids = get_groups(X, sites)
    # plot_pairwise_group(X, y, sites, group_ids, "group_distribution")
    # plot_pairwise_group(X_harmonized, y, sites, group_ids, "group_distribution_combat")
    

    