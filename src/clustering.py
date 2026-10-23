from .gaussian_mixture import GaussianMixtureMAR
import torch
from sklearn.cluster import KMeans

class Clustering(GaussianMixtureMAR):

    def __init__(self,
                 X,
                 M,
                 random_state=123,
                 n_clusters=2,
                 k_range=None,
                 criterion="bic", 
                 cov_type='uni_diag', 
                 max_iter=100, 
                 tol=0.000001, 
                 eps=0.000001, 
                 reg_covar=0.000001, 
                 n_init=20, 
                 _k_tol=1, 
                 device="cpu", 
                 verbose=False):
        super().__init__(k_range, criterion, cov_type, max_iter, tol, eps, reg_covar, n_init, _k_tol, device, verbose)
        self.n_clusters = n_clusters
        self.model = self.fit(X, M)
        self.random_state = random_state

    def _cluster_centers(self):
        """
        Cluster Gaussian centers using train data
        """
        mu_np = self.model.best_model.mu.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
        return kmeans.fit_predict(mu_np)

    def cluster_through_centers(self, X, M):
        idx = self.which_cluster(X, M) # (N, )
        self.center_labels = self._cluster_centers() # (K, )
        return np.array([self.center_labels[idx[i]].item() for i in range(len(idx))])
        # for i-th observation, idx[i] gives the GM label
        # center_labels[idx[i]] gives its final label


    # def _top_k_components(self, X, M=None, k=1):
    #     """
    #     Find which cluster each observation is mostly likely from
    #     """
    #     if M is None:
    #         M = torch.ones_like(X, device=X.device)
    #     _, indices = torch.topk(self.log_prob_N_K_D(X, M).sum(dim=2), dim=1, k=k)
    #     return indices

    # def _fit(self, X, M):
    #     self.fit(X, M)
    #     return self.best_model


    # def cluster_by_topk(self, X, M, k=3):
    #     idx = self._top_k_components(X, M, k)
    #     return idx

    # def cluster_by_logprob(self, X, M):
    #     return self.log_prob_x_m(X, M)



"""Hierarchical merging of Gaussian mixture components via the entropy
criterion of Baudry et al. (2010), "Combining Mixture Components for
Clustering", J. Comput. Graph. Stat. 19(2):332-353."""

from dataclasses import dataclass
import numpy as np


def _xlogx(x):
    """Elementwise x * log(x), with the convention 0 * log(0) = 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    positive = x > 0
    out[positive] = x[positive] * np.log(x[positive])
    return out


def soft_entropy(tau):
    """Entropy of a soft clustering: -sum_i sum_k tau_ik * log(tau_ik)."""
    return -_xlogx(tau).sum()


@dataclass
class CombiResult:
    tau: np.ndarray           # (n_samples, n_clusters) merged posteriors
    labels: np.ndarray        # (n_samples,) hard cluster assignment
    merge_matrix: np.ndarray  # (n_clusters, K_initial); tau_merged = tau @ merge_matrix.T
    groups: list              # groups[j] = original component indices in cluster j
    merges: list              # (n_clusters_after, original components in 1st, in 2nd)
    entropies: dict           # {number of clusters: entropy}, useful for an entropy plot


def entropy_combi(tau, n_clusters, labels=None):
    """Merge mixture components down to `n_clusters` clusters.

    Parameters
    ----------
    tau : (n_samples, K) array
        Posterior membership probabilities from the fitted mixture
        (e.g. GaussianMixture.predict_proba). Rows must sum to 1.
    n_clusters : int
        Target number of clusters, 1 <= n_clusters <= K.
    labels : (n_samples,) int array, optional
        Hard labels in 0..K-1 from the original mixture. If given, they are
        remapped through the merges. If omitted, labels are the MAP
        assignment under the merged posteriors, as defined in the paper.
    """
    tau = np.asarray(tau, dtype=float)
    n_samples, K = tau.shape
    if not np.allclose(tau.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Rows of tau must sum to 1.")
    if not 1 <= n_clusters <= K:
        raise ValueError(f"n_clusters must be between 1 and {K}.")

    current = tau.copy()                    # working posterior matrix
    groups = [[k] for k in range(K)]        # original components per current cluster
    merges = []
    entropies = {K: soft_entropy(current)}

    while current.shape[1] > n_clusters:
        n_cur = current.shape[1]
        # sum_i tau_ik * log(tau_ik) for each current column
        col_terms = _xlogx(current).sum(axis=0)

        # Merging columns a and b replaces them by their sum. The entropy
        # decrease is (a+b)log(a+b) - a log a - b log b, summed over samples.
        best_pair, best_decrease = None, -np.inf
        for a in range(n_cur - 1):
            for b in range(a + 1, n_cur):
                merged_term = _xlogx(current[:, a] + current[:, b]).sum()
                decrease = merged_term - col_terms[a] - col_terms[b]
                if decrease > best_decrease:
                    best_pair, best_decrease = (a, b), decrease

        a, b = best_pair
        merges.append((n_cur - 1, list(groups[a]), list(groups[b])))

        current[:, a] += current[:, b]      # cluster a absorbs cluster b
        current = np.delete(current, b, axis=1)
        groups[a] = groups[a] + groups[b]
        del groups[b]
        entropies[n_cur - 1] = soft_entropy(current)

    # Combining matrix: row j has ones for the original components in cluster j.
    merge_matrix = np.zeros((len(groups), K))
    comp_to_cluster = np.zeros(K, dtype=int)
    for j, members in enumerate(groups):
        merge_matrix[j, members] = 1.0
        comp_to_cluster[members] = j

    if labels is None:
        hard_labels = current.argmax(axis=1)
    else:
        hard_labels = comp_to_cluster[np.asarray(labels, dtype=int)]

    return CombiResult(current, hard_labels, merge_matrix, groups, merges, entropies)




# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr

# from rpy2.robjects import ListVector, StrVector


# def _to_r_matrix(X):
#     return ro.r.matrix(
#         ro.IntVector(X.T.flatten().tolist()),
#         nrow=X.shape[0],
#         ncol=X.shape[1],
#     )


# mcl = ListVector(
#     {
#         'z': _to_r_matrix(resp),
#         'G': resp.shape[1]
#     }
# )

# mcl.rclass = StrVector(["Mclust"])

# mcluster = importr("mclust")

# results = mcluster.clustCombi(mcl)

# labels = np.array(results.rx2('classification')[2]) == 1