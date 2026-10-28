import torch
import numpy as np

from .r_funcs import r_ampute
from .converter import Converter

class AmputeMAR:

    def __init__(self, n:int, d:int, n_pat:int, seed:int):

        if n_pat < 1:
            raise ValueError('Number of patterns should be at least 1')

        self.n = n
        self.d = d
        self.n_pat = n_pat # number of patterns
        self.seeds = np.random.SeedSequence(seed).spawn(3)
        self.rng, self.rng_amp, self.rng_eps = [np.random.default_rng(s) for s in self.seeds]
        self.seed_amp = int(self.rng_amp.integers(1, 2**31 - 1))
        self.seed_eps = int(self.rng_eps.integers(1, 2**31 - 1))
        self.patterns = self.sample_patterns()

        converter = Converter()
        self.to_r = converter.to_r
        self.to_python = converter.to_python


    def sample_patterns(self) -> np.array:
        """Sample incomplete missingness patterns once (1 = observed, 0 = missing).
        Last column is always observed; all-ones rows are excluded."""
        while True:
            if self.n_pat is None:
                self.n_pat = int(self.rng.integers(3, max(round(self.n / 100), 3) + 1))

            patterns = self.rng.integers(0, 2, size=(self.n_pat, self.d - 1))
            # last column is always observed
            patterns = np.hstack([patterns, np.ones((self.n_pat, 1), dtype=int)])
            incomplete = patterns[~patterns.all(axis=1)]
            if incomplete.shape[0] > 0:
                return incomplete


    def ampute(self, X:torch.tensor, prop:float = None, miss_mech:str = 'MAR', miss_type: str = 'RIGHT', weights = None) -> tuple:
        """
        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Complete data.

        prop : float, optional
            Proportion of cases with missing values (so 1 - prop stay complete).
            Default: n_pats / (n_pats + 1), i.e. the complete cases are as
            frequent as each incomplete pattern.

        miss_type : str
            RIGHT:
            ...

        Returns
        -------
        X_NA : torch.Tensor
            Data with NaN values.
        M : torch.Tensor
            Observation mask: 1 = observed, 0 = missing.
        """

        n_pats = len(self.patterns)

        if prop is None:
            prop = n_pats / (n_pats + 1)

        if n_pats > 0:
            result = r_ampute(
                self.to_r(X),
                patterns=self.to_r(self.patterns),
                prop=prop,              # freq defaults to equal across patterns
                mech=miss_mech,
                bycases=True,
                type=self.to_r(miss_type),
                weights=self.to_r(weights),
                seed=self.seed_amp
            )
            # amp = amp.to_numpy() if hasattr(amp, "to_numpy") else np.asarray(amp)
            X_NA = torch.tensor(self.to_python(self.to_python(result)['amp'].to_numpy()), device=X.device, dtype=X.dtype)   # (n, d)

        M = ~torch.isnan(X_NA)

        return X_NA, M
    

    def ampute_eps(self, X:torch.tensor, c0:float = 0.01, eps:float = None):
        # mask1: complete case
        complete = self.rng_eps.random(self.n) < c0
        # mask2: mnar
        mnar = ~complete & (self.rng_eps.random(self.n) < eps / (1 - c0))   # gives eps*n MNAR rows on average
        # mask3: mar
        mar = ~complete & ~mnar

        X_NA = X.clone().detach().cpu().numpy()
        rpatterns = self.to_r(self.patterns)
        for mask, mech, s in zip([mnar, mar], ['MNAR', 'MAR'], [self.seed_eps, self.seed_eps+1]):
            if mask.sum() == 0:
                continue
            result = r_ampute(
                self.to_r(X_NA[mask]),
                patterns = rpatterns,
                mech = mech,
                seed = s
                )
            X_NA[mask] = self.to_python(result)['amp']

        X_NA = torch.tensor(X_NA, device=X.device, dtype=X.dtype)

        M = (~torch.isnan(X_NA)).int()

        return X_NA, M, complete, mnar, mar