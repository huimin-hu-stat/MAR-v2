from .gaussian_mixture import GaussianMixtureMAR
from .propensity_score_estimator import PropensityScoreEstimator
import torch
import numpy as np
from sklearn.linear_model import LinearRegression

class ParametricEstimation(PropensityScoreEstimator):

    def __init__(self,
                 X,
                 M,
                 n_sim=None,
                 k_range=(0,20),
                 cov_type='uni_diag',):
        PropensityScoreEstimator.__init__(self, X=X, M=M, model_k=k_range, cov_type=cov_type)

        self.X = X
        self.M = M
        if n_sim is None:
            self.n_sim = self.X.shape[0]
        else:
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

    def linear_regression(self, Ycol, method, eps=1e-3, propensity_score=None):
        if method in ["generation", "imputation", "complete", "complete-case"]:
            if method == "generation":
                XYtilde = self.model.sample(self.n_sim)[0]
            elif method == "imputation":
                XYtilde = self.model.impute(self.X, self.M)
            elif method == "complete":
                XYtilde = self.X
            elif method == "complete-case":
                XYtilde = self.Xc
            
            Xtilde = XYtilde[:, np.arange(XYtilde.shape[1]) != Ycol % XYtilde.shape[1]]
            Ytilde = XYtilde[:, Ycol]
            
            lin_reg = LinearRegression().fit(Xtilde, Ytilde)
            return lin_reg

        elif method in ["gmm_ipw", "nmmipw", "nmmipw_R", "ipw"]:
            if method == "ipw":
                if propensity_score == None:
                    raise ValueError("propensity_score must be given if method is 'ipw'")
            else:
                if propensity_score != None:
                    raise ValueError("propensity_score can only be given if method is 'ipw'")
                
                if method == "gmm_ipw":
                    propensity_score = self._propensity_score_Xc("gmm", eps=eps)
                elif method == "nmmipw":
                    propensity_score = self._propensity_score_Xc("nmm", eps=eps, fast=True)
                elif method == "nmmipw_R":
                    propensity_score = self._propensity_score_Xc("nmm", eps=eps, fast=False)

            # extract fully observed rows and fit weighted regression
            Xm0 = self.Xc[:, np.arange(self.Xc.shape[1]) != Ycol % self.Xc.shape[1]]
            Ym0 = self.Xc[:, Ycol]
            lin_reg = LinearRegression().fit(Xm0, Ym0, sample_weight=1/propensity_score)
            return lin_reg
            
        else:
            raise NotImplementedError(f"The method {method} was not implemented for linear regression.")

    # def get_truth(self, func):
    #     return func(self.sample_truth(self.n, self.d)).item()

    # mean, quantile, estimate unchanged from the previous version