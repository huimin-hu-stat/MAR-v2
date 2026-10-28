import torch
from torch.distributions import Normal, StudentT

class XUTransform:

    def __init__(self, distr:str = 'normal'):
        distr = distr.casefold()

        if distr == "uni":
            self.x_to_u = lambda x: x
            self.u_to_x = lambda x: x
        elif distr == "logistic":
            self.x_to_u = self._x_to_u_logistic
            self.u_to_x = self._u_to_x_logistic
        elif distr == "normal":
            self.x_to_u = self._x_to_u_normal
            self.u_to_x = self._u_to_x_normal
        elif distr == "t":
            self.x_to_u = self._x_to_u_t
            self.u_to_x = self._u_to_x_t
        elif distr == 'gaussian_mixture':
            self.x_to_u = self._x_to_u_gaussian_mixture
        else:
            raise NotImplementedError


    def _u_to_x_logistic(self, U: torch.tensor) -> torch.tensor:
        eps = torch.finfo(U.dtype).eps
        U = torch.clamp(U, eps, 1-eps)
        return torch.log(U/(1-U))

    def _x_to_u_logistic(self, X:torch.tensor) -> torch.tensor:
        return torch.sigmoid(X)

    def _u_to_x_normal(self, U: torch.tensor) -> torch.tensor:
        return Normal(0, 1).icdf(U)
    
    def _x_to_u_normal(self, X:torch.tensor) -> torch.tensor:
        return Normal(0,1).cdf(X)

    def _u_to_x_t(self, U: torch.tensor) -> torch.tensor:
        return StudentT(self.df).icdf(U)

    def _x_to_u_t(self, X:torch.tensor) -> torch.tensor:
        return StudentT(self.df).cdf(X)

    def _x_to_u_gaussian_mixture(self, X:torch.tensor) -> torch.tensor:
        normal = Normal(
            self.mu.unsqueeze(0),       # (1, K, d)
            self.sigma.unsqueeze(0)     # (1, K, d)
        )
        cdf = normal.cdf(X.unsqueeze(1))  # (N, K, d)
        return self.pi.unsqueeze(1).transpose(1, 2).bmm(cdf).squeeze(1)

