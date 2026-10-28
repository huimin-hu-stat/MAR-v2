from .motivating_example import MotivatingExample
from .converter import Converter
from .gaussian_mixture import GaussianMixtureMAR
from .r_funcs import nmm, predict_prop_nmmipw

import torch
import numpy as np
import nlopt
from scipy.stats import norm


def nmm_ipw_fast(X, O):
    """
    Created by Claude AI!
    Python version of NMMIPW::nmm_preprocessing + NMMIPW::PS (only the 'prop' output).
    X: array (n x K) with NaN for missing entries
    O: array (n x K) with 1 = observed, 0 = missing
    Returns propensities of the fully observed rows (same as ps$prop in R).
    """
    X = np.asarray(X, dtype=float)
    O = np.asarray(O).astype(int)
    N, K = X.shape

    # ---- preprocessing (same as nmm_preprocessing) ----
    code = ((1 - O) * (2 ** np.arange(K))).sum(axis=1)       # R - 1
    uniq = np.unique(code)
    R = np.searchsorted(uniq, code) + 1                      # 1..M, 1 = fully observed
    pattern = 1 - ((uniq[:, None] >> np.arange(K)) & 1)      # M x K, 1 = observed
    M = len(uniq)
    if not np.all(pattern[0] == 1):
        raise ValueError("No fully observed rows: NMMIPW cannot be used.")

    Xz = np.nan_to_num(X, nan=0.0)                           # NA -> 0, like R
    idx1 = np.where(R == 1)[0]                               # complete cases
    sizes = [int(pattern[m].sum()) + 1 for m in range(1, M)]
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    A, B = [], []        # design matrices: rows of pattern m / complete-case rows
    for m in range(1, M):
        cols = pattern[m] == 1
        full = np.hstack([np.ones((N, 1)), Xz[:, cols]])
        A.append(full[R == m + 1])
        B.append(full[idx1])

    def link(x, s):                                          # same as link_func
        t = np.exp(-np.abs(x))
        t = t / (1.0 + t)
        return np.where(x <= 0, t * (1.0 - s) / 0.5, 1.0 - t * s / 0.5)

    n_par = int(offs[-1])
    x0 = np.zeros(n_par)
    for j in range(M - 1):
        x0[offs[j]] = np.log(1.0 / M)                        # same start values as R

    results = []
    for s in 10.0 ** (-np.arange(5, 10)):                    # sigma_star candidates
        best = {"x": None, "f": np.inf}

        def obj(x, grad):
            g = 0.0
            tmp = np.zeros(len(idx1))
            for j in range(M - 1):
                b = x[offs[j]:offs[j + 1]]
                g += np.log(link(A[j] @ b, s)).sum()
                tmp += link(B[j] @ b, s)
            g += np.log(np.maximum(1.0 - tmp, s)).sum()
            f = -g / N
            if f < best["f"] and np.all(tmp - 1.0 + s <= 1e-8):   # remember best feasible point
                best["f"], best["x"] = f, np.array(x)
            return float(f)

        def cons(result, x, grad):
            tmp = np.zeros(len(idx1))
            for j in range(M - 1):
                tmp += link(B[j] @ x[offs[j]:offs[j + 1]], s)
            result[:] = tmp - 1.0 + s

        opt = nlopt.opt(nlopt.LN_COBYLA, n_par)
        opt.set_min_objective(obj)
        opt.set_lower_bounds(-4.0 * np.ones(n_par))
        opt.set_upper_bounds(4.0 * np.ones(n_par))
        opt.add_inequality_mconstraint(cons, 1e-8 * np.ones(len(idx1)))
        opt.set_xtol_rel(1e-4)
        opt.set_maxeval(100000)
        try:
            xs = opt.optimize(x0)
            fs = opt.last_optimum_value()
        except Exception:                                    # e.g. "roundoff-limited": R still returns a result
            if best["x"] is None:
                raise
            xs, fs = best["x"], best["f"]
        results.append((fs, s, xs))

    fs, sigma_star, para = min(results, key=lambda r: r[0])  # like which.min

    tmp = np.zeros(len(idx1))
    for j in range(M - 1):
        tmp += link(B[j] @ para[offs[j]:offs[j + 1]], sigma_star)
    return 1.0 - tmp




class PropensityScoreEstimator(Converter):
    """
    Base model for estimating Propensity Score, X and M are required

    MotivatingExamplePSE is specifically for estimating and comparing PS of data from motivating examples
    """

    def __init__(self, 
                 X: torch.tensor = None,
                 M: torch.tensor = None,
                 model_k: tuple = (0,10),
                 cov_type: str = 'uni_diag'
                 ):
        Converter.__init__(self)

        if X is None or M is None:
            raise ValueError("Missing input: X or M is None.")

        # Fit the main model -- k_range
        self.model = GaussianMixtureMAR(k_range=model_k, cov_type=cov_type)
        self.model.fit(X, M)

        self.X = X
        self.M = M
        self.model_k = model_k
        self.cov_type = cov_type


    def _log_prob_xp_cond_m(self, Xtest, m):
        """
        Compute/estimate logP(Xm=xm|M=m)
        """
        # Extract X^(m) for some m and keep only the cols observed
        Xp = self.X[(self.M==m).all(dim=1)][:, m.bool()]
        # Fit a complete data GMM
        Mp = torch.ones_like(Xp)
        g = GaussianMixtureMAR(k_range=self.model_k, cov_type=self.cov_type)
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
        return ((self.M == m).all(dim=1)).sum() / len(self.X)


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
        if Xtest.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Invalid input dimension: Xtest must be the same dimension of Xtrain, got shape {Xtest.shape}."
            )
        m = torch.ones(self.X.shape[1], device=self.X.device)
        return self._estimate(Xtest, m)

    
    def estimate_propensity_score_nmm(self, Xtest, fast=True):
        """
        Compute the propensity score using NMMIPW:prop_compute
        """
        if Xtest.shape[1] != self.X.shape[1]:
                    raise ValueError(
                        f"Invalid input dimension: Xtest must be the same dimension of Xtrain, got shape {Xtest.shape}."
                    )
        # Python implementation of NMMIPW, only for computing on the train!
        if fast:
            print('Fast method only supports Xtrain')
            return nmm_ipw_fast(X=self.X, O=self.M)

        # NMMIPW:prop_compute()
        rX = self.to_r(self.X)
        rM = self.to_r(self.M)
        rXt = self.to_r(Xtest)
        pre = nmm.nmm_preprocessing(rX, rM)
        ps = nmm.PS(rX, pre.rx2('R'), pre.rx2('pattern'))
        prop_score = predict_prop_nmmipw(ps, rXt, pre.rx2('pattern'))
        return torch.tensor(prop_score, device=self.X.device, dtype=self.X.dtype)




class MotivatingExamplePSE(MotivatingExample, PropensityScoreEstimator):

    def __init__(self,
                 type: str = 'mar', 
                 n: int = 5000, 
                 d: int =2, 
                 alpha: float = 0.7, 
                 model_k: tuple = (0, 10),
                 cov_type: str = 'uni_diag'
                 ):
        MotivatingExample.__init__(self, type, n, d, alpha, model_k)
        X = self.get_X()
        M = self.get_M(X)
        
        PropensityScoreEstimator.__init__(self, X=X, M=M, model_k=model_k, cov_type=cov_type)

    def get_Xtest_cond_x1(self):
        """
        Randomly sample one realization of X1, then condition on X1 to obtain X2
        (for the MAR motivating example only)
        """
        X = torch.empty_like(self.X)
        # X1 is constant across the rows
        X[:, 0] = torch.distributions.Normal(0,1).sample() * torch.ones(self.n, device=self.device)
        # X2|X1
        X[:, 1]= self._conditional(X[:, 0]) # method in motivating example to simulat X2|X1

        if self.d == 3:
            X[:, 2] = torch.distributions.Normal(0,1).sample(self.n)
        return X

    def true_score_motivating_example_mar(self, Xtest):
        """
        The true propensity score only for the MAR motivating example in 2D
        P(M=0|X=x) = (phi(x1) + phi(x2)) / 3
        """
        X1 = Xtest[:,0]
        X2 = Xtest[:,1]
        return (norm.cdf(X1) + norm.cdf(X2)) / 3


    def estimate_propensity_score_gmm_mar(self, Xtest):
        """
        Normalize by W=1
        """
        m = torch.ones(self.d, device=self.device)
        raw_score = self._estimate(Xtest, m)
        
        m1 = m.clone()
        m2 = m.clone()
        m1[0] = 0
        m2[1] = 0
        
        W = raw_score + self._estimate(Xtest, m1) + self._estimate(Xtest, m2)
        
        return raw_score / W