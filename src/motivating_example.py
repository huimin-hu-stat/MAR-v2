from .gaussian_mixture import GaussianMixtureMAR
from .metrics import energy_distance, energy_distance_faster
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.distributions import MultivariateNormal, Normal, Categorical

class MotivatingExample:

    def __init__(self, type='mar', n=5000, d=2, alpha=0.7, model_k=(0,10), p=0.1, B=100, n_jobs=15, device='cpu', dtype=torch.float32):
        self.type = type
        self.n = n
        self.d = d
        self.alpha = alpha
        self.model_k = model_k
        self.p = p
        self.B = B
        self.n_jobs = n_jobs
        self.device = device
        self.dtype = dtype

        get_X_funcs = {
            'mar': self._get_X_mar,
            'mar_like': self._get_X_mar,
            'mnar': self._get_X_mnar
        }
        self.get_X = get_X_funcs[type]

        evaluate_funcs = {
            'mar': self.evaluate_mar,
            'mar_like': self.evaluate_mar,
            'mnar': self.evaluate_eps_mnar
        }
        self.evaluate = evaluate_funcs[type]

    def _get_X_mar(self):
        loc = torch.zeros(self.d, device=self.device, dtype=self.dtype)
        cov = torch.eye(self.d, device=self.device, dtype=self.dtype)
        cov[0, 1] = self.alpha
        cov[1, 0] = self.alpha
        return MultivariateNormal(loc, cov).sample((self.n,))

    def _get_X_mnar(self):
        loc = torch.zeros(self.d, device=self.device, dtype=self.dtype)
        cov = torch.eye(self.d, device=self.device, dtype=self.dtype)
        return MultivariateNormal(loc, cov).sample((self.n,))

    def _X_to_U(self, X):
        norm = Normal(0,1)
        x1 = X[:, 0]
        x2 = X[:, 1]
        u1 = norm.cdf(x1)
        u2 = norm.cdf(x2)
        return u1, u2

    def get_M(self, X, eps=None):
        get_M_probs = {
            'mar': self._get_mar_p,
            'mar_like': self._get_mar_like_p,
            'mnar': self._get_eps_mnar_p
        }
        probs = get_M_probs[self.type](X, eps)
        idx = Categorical(probs).sample()
        masks = torch.stack(
            [
                torch.ones(self.d, device=self.device),
                torch.arange(self.d, device=self.device) != 1,
                torch.arange(self.d, device=self.device) != 0,
            ]
        )
        return masks[idx]

    def _get_mar_p(self, X, eps=None):
        u1, u2 = self._X_to_U(X)
        return torch.stack([u1+u2, 2-u1, 1-u2], dim=1) / 3
        

    def _get_mnar_p(self, X, eps=None):
        u1, u2 = self._X_to_U(X)
        return torch.stack([u1+u2, 2-u2, 1-u1], dim=1) / 3

    def _get_eps_mnar_p(self, X, eps=0.0):
        p1 = self._get_mar_p(X)
        p2 = self._get_mnar_p(X)
        mar = torch.bernoulli(
            torch.full((self.n,), 1 - eps, device=self.device)
        ).bool()
        return torch.where(mar.unsqueeze(1), p1, p2)

    def _get_mar_like_p(self, X, eps=None):
        """
        p((1,1,1)|x)=0.5, p((1,0,1)|x)=0.25 - 0.5 * (1-u1) * (1-u2), p((0,1,1)|x)=0.25 + 0.5 * (1-u1) * (1-u2)
        """
        u1, u2 = self._X_to_U(X)
        p1 = torch.ones_like(u1, device=self.device, dtype=self.dtype) * 0.5
        prod = (u1 - 0.5) * (u2 - 0.5) * 0.5
        probs = torch.stack((p1, 0.25-prod, 0.25+prod), dim=1)
        return probs

    def _get_Xtildes(self, X, M):
        model = GaussianMixtureMAR(self.model_k)
        model.fit(X, M)
        return model.sample(self.n)[0], model.impute(X, M)
    
    def _conditional(self, x):
        mean = self.alpha * x
        std = torch.sqrt(torch.tensor(1.0 - self.alpha**2, device=x.device))
        return torch.distributions.Normal(mean, std).sample()

    def _get_conditional_Xtest(self, X, M):
        """
        p0(xmb|xm)
        """
        Xt1 = X.clone()
        Xt2 = X.clone()
        Xt1[:,0] = self._conditional(X[:,1])
        Xt2[:,1] = self._conditional(X[:,0])
        cond1 = (M==1).all(dim=1)[:, None] # all observed
        cond2 = (M==torch.tensor([0,1,1], device=M.device)).all(dim=1)[:, None] # X1 missing
        return torch.where(cond1, X, torch.where(cond2, Xt1, Xt2))

    def _evaluate(self, eps=None):
        X = self.get_X()
        Xtest = self.get_X()
        M = self.get_M(X, eps=eps)
        Xtilde1, Xtilde2 = self._get_Xtildes(X, M)
        qx1_gen = np.quantile(Xtilde1[:,0], self.p).item()
        qx1_imp = np.quantile(Xtilde2[:,0], self.p).item()
        corr_gen = torch.corrcoef(Xtilde1[:, :2].T)[0, 1].item()
        corr_imp = torch.corrcoef(Xtilde2[:, :2].T)[0, 1].item()
        ed_gen = energy_distance(Xtest, Xtilde1).item()
        ed_imp = energy_distance(Xtest, Xtilde2).item()
        scaled_ed_gen = energy_distance_faster(Xtest, Xtilde1, scale=True).item()
        scaled_ed_imp = energy_distance_faster(Xtest, Xtilde2, scale=True).item()

        return qx1_gen, qx1_imp, corr_gen, corr_imp, ed_gen, ed_imp, scaled_ed_gen, scaled_ed_imp

    def evaluate_mar(self):
        results = np.array(
            Parallel(self.n_jobs)(
                delayed(self._evaluate)()
                for _ in range(self.B)
            )
        )
        results = pd.DataFrame(
            results, 
            columns=['qx1_gen', 'qx1_imp', 'corr_gen', 'corr_imp', 'ed_gen', 'ed_imp', 'scaled_ed_gen', 'scaled_ed_imp']
            )
        return results

    def evaluate_eps_mnar(self, eps_range=np.arange(1.0, step=0.1)):
        return np.array(
            Parallel(self.n_jobs)(
                delayed(self._evaluate)(eps=e)
                for _ in range(self.B)
                for e in eps_range
            )
        ).reshape(self.B, len(eps_range), 8).transpose(1, 0, 2)
