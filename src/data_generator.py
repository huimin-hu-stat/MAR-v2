import torch
from typing import Optional
from torch.distributions import MultivariateNormal
from .ampute_mar import AmputeMAR
from .motivating_example import MotivatingExample
from .xu_transform import XUTransform
import numpy as np


class DataGenerator(XUTransform):

    def __init__(
        self,
        distr: str = "normal",
        mu: torch.tensor = None, # for gaussian mixture
        sigma: torch.tensor = None, # for gaussian mixture
        pi: torch.tensor = None, # for gaussian mixture
        alpha: float = 0.7, # copula
        c: float = 0.01, # positivity
        df: Optional[int] = None,
        device: str = "cpu",
        dtype=torch.float64,
    ):  
        super().__init__(distr=distr)
        self.distr = distr
        self.alpha = alpha
        self.c= torch.tensor(c, dtype=dtype, device=device)
        self.df = df
        self.device = device
        self.dtype = dtype

        self.mu = mu
        self.sigma = sigma
        self.pi = pi

        distr = distr.casefold()

        if distr == "t" and df is None:
            raise ValueError("Student t requires df")
        
        if distr == 'gaussian_mixture' and (mu is None or sigma is None or pi is None):
            raise ValueError("Gassian mixture requires mu, sigma, pi")

        if c > 1:
            raise ValueError("c is too large (> 1)")


    def rand(self, *shape):
        return torch.rand(
            *shape,
            device=self.device,
            dtype=self.dtype
        )


    def _copula_flowgem(
        self,
        n: int,
        d: int
    ):
        """
        Generate dependent uniforms.
            conditional CDF: F(x2|x1) = x2 + alpha*(2*x1-1)*(x2^2 - x2)
            invert this to sample X2 = F_inv(U)
            Inversion yields quadratic equation: alpha*(2*x1-1)*x2^2 + [1 - alpha*(2*x1-1)]*x2 - u = 0
            NOTE: CDF is symmetric so the same method can be used to sample X1 given X2
        """
        U1 = self.rand(n)
        a = self.alpha * (2 * U1 - 1)
        b = 1 - a
        disc = b ** 2 + 4 * a * U1
        U2 = (-b + torch.sqrt(disc)) / (2*a)

        # linear case when a≈0
        U2 = torch.where(torch.abs(a)<1e-10, U1 / b, U2)
        U2 = torch.clamp(U2, 0, 1)

        Urest = self.rand(n, d-2)

        return torch.column_stack([U1, U2, Urest])


    def _ar1_like_covariance(
        self, 
        d: int
    ):
        """
            Generate AR(1) like covariance.
                off-diagonal (except for the last row and column): Sigma_ij = alpha^|i - j|
                NOTE: the last variable is independent from others
        """
        idx = torch.arange(d, device=self.device)
        cov = self.alpha ** (idx[:, None] - idx[None, :]).abs().to(self.dtype)

        # Independent last variable
        cov[-1, :] = 0
        cov[:, -1] = 0
        cov[-1, -1] = 1
        return cov
    
    def _copula_gaussian(
        self, 
        n: int, 
        d: int
    ):
        """
        Sample from a Gaussian copula.

        Args:
            n: number of samples
            d: dimension

        Returns:
            U: (n,d) samples with Uniform(0,1) margins that are correlated via the gaussian copula
        """
        R = self._ar1_like_covariance(d)

        # Cholesky factor such that R = LL^T with L lower-triangular
        L = torch.linalg.cholesky(R)

        # Independent standard normals
        Z = torch.randn(n, d, device=self.device, dtype=self.dtype)

        # Correlated normals
        Z = Z @ L.T

        # Normal CDF -> uniforms
        U = torch.special.ndtr(Z)

        return U


    def _multivariate_gaussian(
        self,
        n: int,
        d: int
    ):
        """
        Sample multivariate Gaussian with AR(1) like covariance
        """
        mean = torch.zeros(d, device=self.device, dtype=self.dtype)
        cov = self._ar1_like_covariance(d=d)
        return MultivariateNormal(mean, cov).sample((n,))


    def _gaussian_mixture(self, n:int) -> torch.tensor:
        """
        Sample Gaussian mixture
        """
        if not torch.isclose(self.pi.sum(), torch.tensor(1.0, device=self.device)):
            self.pi = torch.softmax(self.pi, dim=0)

        # sample component indices
        z = torch.multinomial(self.pi, n, replacement=True)  # (N,)

        # gather parameters for each sample
        mu_z = self.mu[z]        # (N, d)
        sigma_z = self.sigma[z] ** 2  # (N, d)
        cov_z = torch.diag_embed(sigma_z) # (N, d, d)
        return MultivariateNormal(mu_z, cov_z).sample()


    def sample_truth(self, n:int, d:int) -> torch.tensor:
        if self.distr == "normal":
            return self._multivariate_gaussian(n, d)
        
        if self.distr == 'gaussian_mixture':
            return self._gaussian_mixture(n)

        U = self._copula_gaussian(n, d)
        X = self.u_to_x(U)

        return X



class MissingDataGenerator(DataGenerator, AmputeMAR, MotivatingExample):

    def __init__(self, 
                 n:int, d:int,
                 distr = "normal", mu = None, sigma = None, pi = None, alpha = 0.7, c = 0.01, df = None, device = "cpu", dtype=torch.float64):
        DataGenerator.__init__(self, distr=distr, mu=mu, sigma=sigma, pi=pi, alpha=alpha, c=c, df=df, device=device, dtype=dtype)
        AmputeMAR.__init__(self)
        MotivatingExample.__init__(self)

        self.X = self.sample_truth(n, d)
        self.n = n
        self.d = d


    def get_M_ampute(self, n_pat, prop=None, seed=123):
        rng = np.random.default_rng(seed)
        patterns = AmputeMAR.sample_patterns(self.d, self.n, npattern=n_pat, rng=rng)
        Xmis, M = AmputeMAR().ampute(self.X, patterns=patterns, prop=prop)
        return Xmis, M


    def get_M_first2cols(self, eps):
        return MotivatingExample.get_M(self.X, eps=eps)