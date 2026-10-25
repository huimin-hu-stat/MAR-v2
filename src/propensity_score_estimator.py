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

# R function used to obtain propensity score with NMMIPW:::prop_compute
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

    """
    Inherit two classes

    MotivatingExample: data generation and relevant methods for ME

    Converter: R-to-python two-way conversion
    """

    def __init__(self, 
                 X=None, 
                 M=None, 
                 type='mar', 
                 n=5000, 
                 d=3, 
                 model_k=(0, 10), # k_range for fitting the GaussianMixtureMAR
                 device='cpu', 
                 dtype=torch.float32):
        MotivatingExample.__init__(self, type=type, n=n, d=d, model_k=model_k, device=device, dtype=dtype)
        Converter.__init__(self)

        # Simulate motivating example
        if (X is None) & (M is None):
            self.X = self.get_X()
            self.M = self.get_M(self.X)
        else: # Use any MAR (X, M)
            self.X = X
            self.M = M

        # Fit the main model -- k_range 
        self.model = GaussianMixtureMAR(k_range=model_k)
        self.model.fit(self.X, self.M)

    def get_Xtest_cond_x1(self):
        """
        Randomly sample one realization of X1, then condition on X1 to obtain X2
        (for the MAR motivating example only)
        """
        if len(self.X) > 3:
            return NotImplemented

        X = torch.empty_like(self.X)
        # X1 is constant across the rows
        X[:, 0] = torch.distributions.Normal(0,1).sample() * torch.ones(self.n, device=self.device)
        # X2|X1
        X[:, 1]= self._conditional(X[:, 0]) # method in motivating example to simulat X2|X1

        if len(self.X) == 3:
            X[:, 2] = torch.distributions.Normal(0,1).sample(self.n)

        return X

    def _log_prob_xp_cond_m(self, Xtest, m):
        """
        Compute/estimate logP(Xm=xm|M=m)
        """
        # Extract X^(m) for some m and keep only the cols observed
        Xp = self.X[(self.M==m).all(dim=1)][:, m.bool()]
        # Fit a complete data GMM
        Mp = torch.ones_like(Xp, device=Xp.device)
        g = GaussianMixtureMAR(k_range=(0,10))
        g.fit(Xp, Mp)
        return g.log_prob_x(Xtest[:, m.bool()]) # Estimate the log density using the density estimate of partial data

    def _log_prob_xp(self, Xtest, m):
        """
        Compute/estimate logP(X=x) only for the observed columns (Xm)
        P(M=m|Xm) = P(Xm|M=m)P(M=m)/P(Xm)
        """
        return self.model.log_prob_x(Xtest, cols=torch.where(m==1))

    def _phat_M(self, m):
        """
        Compute/estimate P(M=m)
        """
        return ((self.M == m).all(dim=1)).sum() / self.n

    def _estimate(self, Xtest, m):
        """
        Compute/estimate the missingness mechanism P(M=m|Xm=xm)
        """
        # exp(lp1 - lp2) = P(Xm=xm|M=m) / P(X=x, M=0)
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
        """
        Compute propensity score using direction method
        (... indirect: subtraction)
        """
        m = torch.ones(self.d, device=self.device)
        return self._estimate(Xtest, m)


    def true_score_motivating_example_mar(self, Xtest):
        """
        The true propensity score only for the MAR motivating example in 2D
        P(M=0|X=x) = (phi(x1) + phi(x2)) / 3
        """
        X1 = Xtest[:,0]
        X2 = Xtest[:,1]
        return (norm.cdf(X1) + norm.cdf(X2)) / 3

    
    def estimate_propensity_score_nmm(self, Xtest):
        """
        Compute the propensity score using NMMIPW:prop_compute
        """
        rX = self.to_r(self.X)
        rM = self.to_r(self.M)
        rXt = self.to_r(Xtest)
        pre = nmm.nmm_preprocessing(rX, rM)
        ps = nmm.PS(rX, pre.rx2('R'), pre.rx2('pattern'))
        prop_score = predict_prop_nmmipw(ps, rXt, pre.rx2('pattern'))
        return np.array(prop_score)


