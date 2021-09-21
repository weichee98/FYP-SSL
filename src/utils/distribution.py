import os.path as osp
import sys
import numpy as np
import networkx as nx
from enum import Enum
from scipy.stats import gaussian_kde

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.data import corr_mx_flatten


class SiteDistribution:

    class METRIC(Enum):
        KL = 0
        JS = 1
        HELLINGER = 2

        @staticmethod
        def _hellinger_hist(p, q):
            assert p.shape == q.shape
            h = np.sqrt(p) - np.sqrt(q)
            h = np.linalg.norm(h) / np.sqrt(2)
            return h

        @staticmethod
        def _hellinger_gauss(p_mu, p_var, q_mu, q_var):
            e = -0.25 * (p_mu - q_mu) ** 2 / (p_var + q_var)
            m = np.sqrt(np.sqrt(4 * p_var * q_var) / (p_var + q_var))
            h2 = 1 - m * np.exp(e)
            return np.sqrt(h2)

        @staticmethod
        def _js_hist(p, q, eps=1e-5):
            assert p.shape == q.shape
            p, q = np.maximum(p, eps), np.maximum(q, eps)
            m = 0.5 * (p + q)
            kl_pm = p * np.log(p / m)
            kl_qm = q * np.log(q / m)
            js = 0.5 * (kl_pm + kl_qm)
            return np.sum(js)

        @staticmethod
        def _kl_hist(p, q, eps=1e-5):
            assert p.shape == q.shape
            p, q = np.maximum(p, eps), np.maximum(q, eps)
            kl_div = p * np.log(p / q)
            return np.sum(kl_div)

        @staticmethod
        def _kl_gauss(p_mu, p_var, q_mu, q_var):
            kl = 0.5 * (
                np.log(q_var) - np.log(p_var) + \
                (p_var + (p_mu - q_mu) ** 2) / q_var - 1
            )
            return kl

    class METHOD(Enum):
        GAUSS = 0
        HIST = 1
        KDE = 2

        @staticmethod
        def _get_mean_var(site_mean):
            return dict(
                (s, (np.mean(v), np.var(v)))
                for s, v in site_mean.items()
            )

        @staticmethod
        def _get_histogram(site_mean, bins=1000, vmin=-1, vmax=1):
            site_hist = dict(
                (s, np.histogram(v, bins=bins, range=(vmin, vmax))[0])
                for s, v in site_mean.items()
            )
            site_hist = dict(
                (s, v / np.sum(v)) for s, v in site_hist.items()
            )
            return site_hist

        @staticmethod
        def _get_binned_kde(site_mean, bins=1000, vmin=-1, vmax=1):
            kde = dict(
                (s, gaussian_kde(v))
                for s, v in site_mean.items()
            )
            x = np.linspace(vmin, vmax, bins)
            site_hist = dict(
                (s, v.pdf(x)) for s, v in kde.items()
            )
            site_hist = dict(
                (s, v / np.sum(v)) for s, v in site_hist.items()
            )
            return site_hist
    
    @staticmethod
    def get_site_mean(X, sites, fisher=False):
        X = corr_mx_flatten(X)
        if fisher:
            X = np.arctanh(X)
        group_dic = {}
        for site in np.unique(sites):
            idx = np.argwhere(sites == site).flatten()
            group_mean = np.mean(X[idx], axis=0)
            if fisher:
                group_mean = np.tanh(group_mean)
            group_dic[site] = group_mean
        return group_dic

    def distribution_heatmap(self, X, sites, metric, method, **kwargs):
        hist_kwg = dict(bins=kwargs["bins"]) if "bins" in kwargs else dict()

        site_mean = self.get_site_mean(X, sites, fisher=True)
        if method == self.METHOD.GAUSS:
            site_mu_var = self.METHOD._get_mean_var(site_mean)
        elif method == self.METHOD.HIST:
            site_hist = self.METHOD._get_histogram(site_mean, **hist_kwg)
        elif method == self.METHOD.KDE:
            site_kde = self.METHOD._get_binned_kde(site_mean, **hist_kwg)
        else:
            raise NotImplementedError(
                "method {} is not implemented".format(method)
            )

        unique_sites = np.unique(sites)
        dist_diff = dict((s1, dict()) for s1 in unique_sites)
        num_sites = len(unique_sites)

        for i in range(num_sites):
            for j in range(i, num_sites):
                s1, s2 = unique_sites[i], unique_sites[j]
                if s1 == s2:
                    dist_diff[s1][s2] = 0
                    continue

                if method == self.METHOD.GAUSS:
                    if metric == self.METRIC.HELLINGER:
                        d1 = d2 = self.METRIC._hellinger_gauss(*site_mu_var[s1], *site_mu_var[s2])
                    elif metric == self.METRIC.KL:
                        d1 = self.METRIC._kl_gauss(*site_mu_var[s1], *site_mu_var[s2])
                        d2 = self.METRIC._kl_gauss(*site_mu_var[s2], *site_mu_var[s1])
                    else:
                        raise NotImplementedError(
                            "{} metric not implemented for {} method"
                            .format(metric, method)
                        )

                elif method == self.METHOD.HIST:
                    if metric == self.METRIC.HELLINGER:
                        d1 = d2 = self.METRIC._hellinger_hist(site_hist[s1], site_hist[s2])
                    elif metric == self.METRIC.JS:
                        d1 = d2 = self.METRIC._js_hist(site_hist[s1], site_hist[s2])
                    elif metric == self.METRIC.KL:
                        d1 = self.METRIC._kl_hist(site_hist[s1], site_hist[s2])
                        d2 = self.METRIC._kl_hist(site_hist[s2], site_hist[s1])
                    else:
                        raise NotImplementedError(
                            "{} metric not implemented for {} method"
                            .format(metric, method)
                        )

                elif method == self.METHOD.KDE:
                    if metric == self.METRIC.HELLINGER:
                        d1 = d2 = self.METRIC._hellinger_hist(site_kde[s1], site_kde[s2])
                    elif metric == self.METRIC.JS:
                        d1 = d2 = self.METRIC._js_hist(site_kde[s1], site_kde[s2])
                    elif metric == self.METRIC.KL:
                        d1 = self.METRIC._kl_hist(site_kde[s1], site_kde[s2])
                        d2 = self.METRIC._kl_hist(site_kde[s2], site_kde[s1])
                    else:
                        raise NotImplementedError(
                            "{} metric not implemented for {} method"
                            .format(metric, method)
                        )

                else:
                    raise NotImplementedError(
                        "method {} is not implemented".format(method)
                    )

                dist_diff[s1][s2] = d1
                dist_diff[s2][s1] = d2

        return dist_diff

    def get_site_grouping(self, X, sites, metric, method, threshold, **kwargs):
        """
        Partition sites into optimal subsets, sites in the
        same subset would have statistical difference less 
        than the specified threshold

        metric: METRIC.KL, METRIC.JS, METRIC.HELLINGER
        method: METHOD.GAUSS, METHOD.HIST, METHOD.KDE

        threshold: the maximum difference in distribution allowed

        kwargs: additional keyword argument to be passed
        - bins: int - if method == METHOD.HIST or METHOD.KDE
        """
        dist_diff = self.distribution_heatmap(X, sites, metric, method, **kwargs)

        """
        Create a graph with nodes as the sites,
        an edge is present between 2 sites if their statistical
        difference is less than the specified threshold
        """
        edge_list = list()
        for s1 in dist_diff:
            for s2 in dist_diff[s1]:
                if s2 >= s1:
                    continue
                if dist_diff[s1][s2] > threshold:
                    continue
                if dist_diff[s2][s1] > threshold:
                    continue
                edge_list.append((s1, s2))

        graph = nx.Graph()
        graph.add_edges_from(edge_list)

        """
        Cliques are strongly connected subgraphs, for example:

        A = [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
        ]
        - nodes 0 and 1 form a clique
        - nodes 2, 3, 4 form a clique

        We will first list out all possible cliques.
        """
        cliques = list(nx.clique.find_cliques(graph))
        site_group = dict((s, None) for s in graph.nodes)
        site_cliques = dict((s, list()) for s in graph.nodes)
        fixed_cliques = set()
        visited_cliques = set()

        """
        First, we will find sites that can only present in 1 clique.
        Then, we will mark the cliques for those sites as visited and 
        fixed. Sites within these cliques will have a fixed group.
        
        If a site can be present in more than 1 clique, the site will 
        be assigned to the largest possible clique.
        """

        for i, c in enumerate(cliques):
            for s in c:
                site_cliques[s].append(i)

        for site in site_cliques:
            if len(site_cliques[site]) > 1:
                continue
            if len(site_cliques[site]) == 0:
                return None
            i = site_cliques[site][0]
            visited_cliques.add(i)
            fixed_cliques.add(i)
            for s in cliques[i]:
                cs = site_group[s]
                if cs is None or \
                    len(cliques[i]) > len(cliques[cs]):
                    site_group[s] = i
        
        def complete_groups(site_group, visited_cliques):
            """
            Here, we will assign the remaining sites to their optimal 
            cliques (groups).

            A set of optimal cliques is defined such that:
            1.  The set contains a minimal number of cliques.
            2.  The cliques must be as disjoint as possible, which
                means minimal intersection between cliques.

            If all sites have been assigned a group, the algorithm
            backtracks and continue to search for other possible
            set of cliques that have lower number of intersection.

            If no possible set of cliques can include all the sites, 
            the algorithm returns None.
            """

            # check if all sites have a group already
            if None not in site_group.values():
                return 0, site_group.copy()

            # keep track of the minimum number of intesections
            min_intersect = float("inf")
            best_site_group = None
            new_site_group = site_group.copy()

            for site, group in site_group.items():
                # if a site has been assigned a group, continue
                # else, visit the possible cliques for that site
                if group is not None:
                    continue
                to_visit = site_cliques[site]
                
                for c in to_visit:
                    # if the clique is visited previously, continue,
                    # else, visit the clique and set the current number
                    # of intersection as 0
                    if c in visited_cliques:
                        continue
                    visited_cliques.add(c)
                    num_intersect = 0

                    # iterate through the involved sites, if a site has not been 
                    # assigned a group, assign the clique directly, else if the 
                    # site has been assigned a fixed group, continue, else, assign
                    # a larger clique to the site,
                    # the number of intersections is incremented by 1 whenever
                    # a site with a group assigned previously is met
                    for s in cliques[c]:
                        cs = new_site_group[s]
                        if cs is None:
                            new_site_group[s] = c
                        else:
                            num_intersect += 1
                            if cs in fixed_cliques:
                                continue
                            if len(cliques[c]) > len(cliques[cs]):
                                new_site_group[s] = c

                    # if the number of intersection after assigning this clique
                    # is less than the minimum intersection keep track so far, 
                    # continue to assign other cliques, else, skip this step
                    if num_intersect < min_intersect:
                        res = complete_groups(new_site_group, visited_cliques)
                        num_intersect += res[0]
                        # add up the current number of intersections and the number 
                        # of intersections after assigning other cliques, if the
                        # total number of intersection is lower than the minimum
                        # number of intersection keep track so far, we have found
                        # a better set of cliques, thus, overwrite the current result
                        # with the better result
                        if num_intersect < min_intersect:
                            min_intersect = num_intersect
                            best_site_group = res[1]

                    # revert the assignment of this clique, and mark this clique
                    # as not visited, then continue to search for other possible
                    # cliques that might be optimal        
                    for s in cliques[c]:
                        new_site_group[s] = site_group[s]
                    visited_cliques.remove(c)

            # return the minimum number of intersection and the best clique assignment
            return min_intersect, best_site_group

        _, best_site_group = complete_groups(site_group, visited_cliques)
        if best_site_group is None:
            return None
        group_subset = dict()
        for site, group in sorted(best_site_group.items()):
            group_subset.setdefault(group, list())
            group_subset[group].append(site)
        return sorted(group_subset.values())


if __name__ == "__main__":
    from ABIDE import *

    X, _ = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()
    groupings = SD.get_site_grouping(X, sites, SD.METRIC.HELLINGER, SD.METHOD.GAUSS, 0.05)
    print(groupings)