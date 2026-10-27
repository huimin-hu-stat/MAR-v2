from .gaussian_mixture import GaussianMixtureMAR
from .propensity_score_estimator import PropensityScoreEstimator
import torch
import numpy as np

class ParametricEstimation(GaussianMixtureMAR, PropensityScoreEstimator):

    def __init__(self,
                 X,
                 M,
                 n_sim=2000,
                 k_range=(0,20),
                 criterion="bic",
                 cov_type='uni_diag',
                 max_iter=100,
                 tol=0.000001,
                 eps=0.000001,
                 reg_covar=0.000001,
                 n_init=20,
                 _k_tol=1,
                 device="cpu",
                 verbose=False,):
        GaussianMixtureMAR.__init__(self, k_range, criterion, cov_type, max_iter, tol, eps,
                         reg_covar, n_init, _k_tol, device, verbose)
        PropensityScoreEstimator.__init__(self, X=X, M=M, model_k=k_range, cov_type=cov_type)

        self.X = X
        self.M = M
        self.n_sim = n_sim
        self.Xc = X[(M==1).all(dim=1)]


    def _propensity_score_Xc(self, method, eps=1e-3, fast=True):
        if method not in ['gmm', 'nmm']:
            raise NotImplementedError
        
        if method == 'gmm':
            ps = self.estimate_propensity_score_gmm(self.Xc)
        if method == 'nmm':
            ps = self.estimate_propensity_score_nmm(self.Xc, fast=fast)

        if not isinstance(ps, torch.Tensor):
            ps = torch.tensor(ps, device=self.X.device, dtype=self.X.dtype)

        return torch.clamp(ps, min=eps)

    
    def mean_by_generation(self):
        Xtilde = self.model.sample(self.n_sim)[0]
        return Xtilde.mean(dim=0)


    def mean_by_ipw(self, propensity_score=None):
        return np.average(self.Xc, axis=0, weights=1/propensity_score)
    

    def quantile_by_generation(self, var_index=0, p=0.1):
        """
        Estimate the p-th quantile of the variable with index var_index
        """
        Xtilde = self.model.sample(self.n_sim)[0]
        x = Xtilde[:, var_index]
        return np.quantile(x, p)


    def quantile_by_ipw(self, var_index=0, p=0.1, propensity_score=None):
        return np.quantile(self.Xc[:, var_index], p, method="inverted_cdf", weights=1/propensity_score)


    def estimate(self, func):
        Xtilde = self.model.sample(self.n_sim)[0]
        return func(Xtilde).item()

    # def get_truth(self, func):
    #     return func(self.sample_truth(self.n, self.d)).item()

    # mean, quantile, estimate unchanged from the previous version