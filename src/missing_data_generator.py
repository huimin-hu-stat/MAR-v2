import torch
from .ampute_mar import AmputeMAR
from .data_generator import DataGenerator
from .motivating_example import UtoM


class MissingDataGenerator(DataGenerator, AmputeMAR, UtoM):

    def __init__(self, 
                 n:int, d:int,
                 n_pat:int, seed:int,
                 distr = "normal", mu = None, sigma = None, pi = None, alpha = 0.7, c = 0.01, df = None, device = "cpu", dtype=torch.float64):
        DataGenerator.__init__(self, distr=distr, mu=mu, sigma=sigma, pi=pi, alpha=alpha, c=c, df=df, device=device, dtype=dtype)
        AmputeMAR.__init__(self, n=n, d=d, n_pat=n_pat, seed=seed)
        UtoM.__init__(self, d=d)
        self.n = n
        self.d = d

    def get_X(self):
        return self.sample_truth(n=self.n, d=self.d)

    def get_M_ampute_eps(self, X, eps):
        Xmis, M, _, _, _ = self.ampute_eps(X, eps=eps)
        return Xmis, M

    def get_M_eps_mar(self, X, eps):
        u1 = self.x_to_u(X[:,0])
        u2 = self.x_to_u(X[:,1])
        M = self.get_M(missing_type='contaminated_mar', u1=u1, u2=u2, eps=eps)
        return M