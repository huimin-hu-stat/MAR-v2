from .motivating_example import MotivatingExample
from .converter import Converter
from .gaussian_mixture import GaussianMixtureMAR
import torch
import numpy as np
from scipy.stats import norm

#=== FOR NMMIPW ===#
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
nmm = importr("NMMIPW")
#==================#

ro.r('''
predict_prop <- function(ps, newdata, pattern) {
  X <- as.matrix(newdata); M <- nrow(pattern)
  work_df <- c(); selection <- c(0)
  for (m in 2:M) {
    work_df <- cbind(work_df, cbind(1, X[, pattern[m, ] == 1]))
    selection <- c(selection, sum(pattern[m, ]) + 1)
  }
  selection <- cumsum(selection)
  NMMIPW:::prop_compute(para = ps$gamma_para, work_df = work_df,
                        R = rep(1, nrow(X)), selection = selection,
                        sigma_star = ps$sigma_star, pattern = pattern)
}
''')
predict_prop_nmmipw = ro.r["predict_prop"]



class PropensityScoreEstimator(MotivatingExample, Converter):

    def __init__(self, X=None, M=None, type='mar', n=5000, d=3, alpha=0.7, model_k=(0, 10), p=0.1, B=100, n_jobs=15, device='cpu', dtype=torch.float32):
        MotivatingExample.__init__(self, type, n, d, alpha, model_k, p, B, n_jobs, device, dtype)
        Converter.__init__(self)

        if (X is None) & (M is None):
            self.X = self.get_X()
            self.M = self.get_M(self.X)
        else:
            self.X = X
            self.M = M

        self.model0 = GaussianMixtureMAR(k_range=(0,10))
        self.model0.fit(self.X, self.M)

    def get_Xtest_cond_x1(self):
        mu = torch.zeros(self.d, device=self.device)
        cov = torch.eye(self.d, device=self.device)
        X = torch.distributions.MultivariateNormal(mu, cov).sample((self.n,))
        X[:, 0] = torch.distributions.Normal(0,1).sample() * torch.ones(self.n, device=self.device)
        X[:, 1]= self._conditional(X[:, 0])
        return X

    def _log_prob_xp_cond_m(self, Xtest, m):
        Xp = self.X[(self.M==m).all(dim=1)][:, m.bool()]
        Mp = torch.ones_like(Xp, device=Xp.device)
        g = GaussianMixtureMAR(k_range=(0,10))
        g.fit(Xp, Mp)
        return g.log_prob_x(Xtest[:, m.bool()])

    def _log_prob_xp(self, Xtest, m):
        return self.model0.log_prob_x(Xtest, torch.where(m==1))

    def _phat_M(self, m):
        return ((self.M == m).all(dim=1)).sum() / self.n

    def _estimate(self, Xtest, m):
        lp1 = self._log_prob_xp_cond_m(Xtest, m)
        lp2 = self._log_prob_xp(Xtest, m)
        p1 = self._phat_M(m)
        return (lp1 - lp2).exp() * p1

    # def estimate_by_subtraction(self, Xtest):
    #     m = torch.ones(self.d, device=self.device)
    #     m1 = m.clone()
    #     m2 = m.clone()
    #     m1[0] = 0
    #     m2[1] = 0
    #     raw_est = 1 - self._estimate(Xtest, m1) - self._estimate(Xtest, m2)
    #     return torch.clamp(raw_est, min=0)

    def estimate_propensity_score_gmm(self, Xtest):
        m = torch.ones(self.d, device=self.device)
        return self._estimate(Xtest, m)


    def true_score_motivating_example_mar(self, Xtest):
        X1 = Xtest[:,0]
        X2 = Xtest[:,1]
        return (norm.cdf(X1) + norm.cdf(X2)) / 3

    
    def estimate_propensity_score_nmm(self, Xtest):
        rX = self.to_r(self.X)
        rM = self.to_r(self.M)
        rXt = self.to_r(Xtest)
        pre = nmm.nmm_preprocessing(rX, rM)
        ps = nmm.PS(rX, pre.rx2('R'), pre.rx2('pattern'))
        prop_score = predict_prop_nmmipw(ps, rXt, pre.rx2('pattern'))
        return np.array(prop_score)


