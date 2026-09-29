import torch

from torch.distributions import Normal, StudentT, Categorical

class MissingnessGenerator:

    def __init__(self, X, type=None, c=None):
        self.X = X
        self.tpye = type
        self.c = c

    def _x_to_u_unif(self):
        return self.X[:,0], self.X[:, 1]

    def _x_to_u_logistic(self):
        return torch.sigmoid(self.X[:,0]), torch.sigmoid(self.X[:,1])

    def _x_to_u_gaussian_mixture(self, mu, sigma, pi):
        normal1 = Normal(mu[:, 0].unsqueeze(1), sigma[:, 0].unsqueeze(1))
        normal2 = Normal(mu[:, 1].unsqueeze(1), sigma[:, 1].unsqueeze(1))
        return pi @ normal1.cdf(self.X[:,0].repeat(len(mu), 1)), pi @ normal2.cdf(self.X[:,1].repeat(len(mu), 1))

    def _x_to_u_normal(self):
        normal = Normal(0,1)
        return normal.cdf(self.X[:,0]), normal.cdf(self.X[:,1])

    def _x_to_u_t(self, df):
        tdist = StudentT(df)
        return tdist.cdf(self.X[:,0]), tdist.cdf(self.X[:,1])

    def _u_to_x_unif(self):
        return self.U[:, 0], self.U[:, 1]

    def _u_to_x_logistic(self):
        return torch.logit(self.U[:, 0]), torch.logit(self.U[:, 1])

    def _u_to_x_normal(self):
        normal = Normal(0, 1)
        return normal.icdf(self.U[:, 0]), normal.icdf(self.U[:, 1])

    def _u_to_x_t(self, df):
        tdist = StudentT(df)
        return tdist.icdf(self.U[:, 0]), tdist.icdf(self.U[:, 1])

    # 3D example
    def example1(self, u1, u2, eps=0.0):
        """
        Generate MAR / MNAR missingness. MAR: eps=0.0
        MAR:
        P((0,0,0)|x) = 1 - P((0,1,0)|x) - P((1,0,0)|x)
        P((0,1,0)|x) = (2 - max(P(x1), c))/3
        P((1,0,0)|x) = (1 - max(P(x2), c))/3
        MNAR:
        P((0,0,0)|x) = 1 - P((0,1,0)|x) - P((1,0,0)|x)
        P((0,1,0)|x) = (2 - max(P(x2), c))/3
        P((1,0,0)|x) = (1 - max(P(x1), c))/3
        """
        n, d = self.X.shape

        G1 = torch.maximum(u1, self.c)
        G2 = torch.maximum(u2, self.c)

        P_MAR = torch.stack(
                    [
                        (G1+G2)/3,
                        (2-G1)/3, # X2 missing
                        (1-G2)/3 # X1 missing
                    ],
                    dim=1
                )
                
        P_MNAR = torch.stack(
            [
                (G1+G2)/3,
                (2-G2)/3, # X2 missing
                (1-G1)/3 # X1 missing
            ],
            dim=1
        )
        
        use_mar = torch.bernoulli(
            torch.full((n,), 1 - eps, device=self.X.device)
            ).bool()

        P = torch.where(use_mar.unsqueeze(1), P_MAR, P_MNAR)
        
        choice = Categorical(P).sample()
        
        masks = torch.stack(
            [
                torch.ones(
                    d,
                    device=self.device
                ), #(1,1,1)

                torch.arange(
                    d,
                    device=self.device
                ) != 1, #(1,0,1)

                torch.arange(
                    d,
                    device=self.device
                ) != 0, #(0,1,1)
            ]
        )

        return masks[choice]

    # 2D example
    def example2(self, u1, u2):
        product = (u1 - 0.5) * (u2 - 0.5) * 0.5
        probs = torch.stack(
            [
                1/2 * torch.ones_like(u1, device=u1.device), # (0,0)
                1/4 + product, # (0,1)
                1/4 - product # (1,0)
            ],
            dim=1
        )
        choice = Categorical(probs).sample()

        masks = torch.stack(
            [
                torch.ones(2, device=u1.device),
                torch.arange(2, device=u1.device) != 1,
                torch.arange(2, device=u1.device) != 0
            ]
        )
        return masks[choice]