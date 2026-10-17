from .motivating_example import MotivatingExample
from .gaussian_mixture import GaussianMixtureMAR
import torch

class PropensityScoreEstimator(MotivatingExample):

    def __init__(self, type='mar', n=5000, d=3, alpha=0.7, model_k=(0, 10), p=0.1, B=100, n_jobs=15, device='cpu', dtype=torch.float32):
        super().__init__(type, n, d, alpha, model_k, p, B, n_jobs, device, dtype)
        self.X = self.get_X()
        self.M = self.get_M(self.X)

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

    def estimate_by_subtraction(self, Xtest):
        m = torch.ones(self.d, device=self.device)
        m1 = m.clone()
        m2 = m.clone()
        m1[0] = 0
        m2[1] = 0
        raw_est = 1 - self._estimate(Xtest, m1) - self._estimate(Xtest, m2)
        return torch.clamp(raw_est, min=0)

    def estimate_direct(self, Xtest):
        m = torch.ones(self.d, device=self.device)
        return self._estimate(Xtest, m)