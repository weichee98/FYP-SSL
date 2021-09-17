import os.path as osp
import sys
from matplotlib.pyplot import get
import numpy as np
import networkx as nx

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.data import corr_mx_flatten


class SiteDistribution:
    
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

    @staticmethod
    def _kl(p, q):
        eps = min(np.min(q[q > 0]), np.min(p[p > 0]), 1e-5)
        p, q = np.maximum(p, eps), np.maximum(q, eps)
        kl_div = p * np.log(p / q)
        return np.sum(kl_div, axis=0)

    @staticmethod
    def _kl_gauss(p_mu, p_var, q_mu, q_var):
        kl = 0.5 * (
            np.log(q_var) - np.log(p_var) + \
            (p_var + (p_mu - q_mu) ** 2) / q_var - 1
        )
        return kl

    def _hist_kldiv(self, a, b, nbins=200):
        ahist = np.histogram(a, bins=nbins, range=(-1, 1))[0]
        bhist = np.histogram(b, bins=nbins, range=(-1, 1))[0]
        ahist = ahist / np.sum(ahist)
        bhist = bhist / np.sum(bhist)
        return self._kl(ahist, bhist)

    def _normal_kldiv(self, p, q):
        p_mu, p_var = np.mean(p), np.var(p)
        q_mu, q_var = np.mean(q), np.var(q)
        return self._kl_gauss(p_mu, p_var, q_mu, q_var)

    def kl_distribution_heatmap(self, X, sites):
        """
        return: dict of dict
            {p -> {q -> kl_divergence}}
        """
        site_mean = self.get_site_mean(X, sites, fisher=True)
        unique_sites = np.unique(sites)
        dist_diff = dict()
        for s1 in unique_sites:
            dist_diff[s1] = dict()
            for s2 in unique_sites:
                if s1 ==  s2:
                    dist_diff[s1][s2] = 0
                else:
                    dist_diff[s1][s2] = self._normal_kldiv(
                        site_mean[s1], site_mean[s2]
                    )
        return dist_diff

    def kl_hist_distribution_heatmap(self, X, sites, bins=1000):
        """
        return: dict of dict
            {p -> {q -> kl_divergence}}
        """
        site_mean = self.get_site_mean(X, sites, fisher=True)
        unique_sites = np.unique(sites)
        dist_diff = dict()
        for s1 in unique_sites:
            dist_diff[s1] = dict()
            for s2 in unique_sites:
                if s1 ==  s2:
                    dist_diff[s1][s2] = 0
                else:
                    dist_diff[s1][s2] = self._hist_kldiv(
                        site_mean[s1], site_mean[s2], bins
                    )
        return dist_diff

    def get_site_grouping(self, X, sites, metric, threshold, **kwargs):
        """
        Partition sites into optimal subsets, sites in the
        same subset would have statistical difference less 
        than the specified threshold

        metric
        - "kl": kl divergence assuming normal distribution
        - "kl_hist": kl divergence using histogram, 1000 bins

        threshold: the maximum difference in distribution allowed

        kwargs: additional keyword argument to be passed
        - bins: int - if metric == "kl_hist"
        """
        if metric == "kl":
            dist_diff = self.kl_distribution_heatmap(X, sites)
        elif metric == "kl_hist":
            if "bins" in kwargs:
                kl_hist_kwg = dict(bins=kwargs["bins"])
            else:
                kl_hist_kwg = dict()
            dist_diff = self.kl_hist_distribution_heatmap(
                X, sites, **kl_hist_kwg
            )
        else:
            raise ValueError("invalid metric {}".format(metric))

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
    import matplotlib.pyplot as plt

    X, _ = load_data_fmri()
    sites = get_sites()
    SD = SiteDistribution()
    groupings = SD.get_site_grouping(X, sites, "kl", 0.01)
    print(groupings)