import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from .converter import Converter


# import os
# from pathlib import Path
# R_HOME = Path(r"C:\Users\hu\AppData\Local\Programs\R\R-4.5.3")
# assert (R_HOME / "bin" / "R.exe").exists(), "R.exe not found"
# assert (R_HOME / "bin" / "x64" / "R.dll").exists(), "R.dll not found"
# os.environ["R_HOME"] = str(R_HOME)
# os.environ["PATH"] = (
#     str(R_HOME / "bin" / "x64")
#     + os.pathsep
#     + str(R_HOME / "bin")
#     + os.pathsep
#     + os.environ["PATH"]
# )


mice = importr("mice")

class AmputeMAR(Converter):

    @staticmethod
    def sample_patterns(d, n, npattern=None, rng=None):
        """Sample incomplete missingness patterns once (1 = observed, 0 = missing).
        Last column is always observed; all-ones rows are excluded."""
        rng = np.random.default_rng() if rng is None else rng
        while True:
            if npattern is None:
                npattern = int(rng.integers(3, max(round(n / 100), 3) + 1))
            #npattern= 10
            patterns = rng.integers(0, 2, size=(npattern, d - 1))
            patterns = np.hstack([patterns, np.ones((npattern, 1), dtype=int)])
            incomplete = patterns[~patterns.all(axis=1)]
            if incomplete.shape[0] > 0:
                return incomplete

    def ampute(self, X, patterns=None, prop=None, rng=None, miss_type=None, weights=None):
        """
        Parameters
        ----------
        X : torch.Tensor, shape (n, d)
            Complete data.
        patterns : array-like, shape (n_pats, d), optional
            Binary patterns: 1 = observed, 0 = missing. All-ones rows are dropped.
            If None, new patterns are sampled.
        prop : float, optional
            Proportion of cases with missing values (so 1 - prop stay complete).
            Default: n_pats / (n_pats + 1), i.e. the complete cases are as
            frequent as each incomplete pattern.
        rng : np.random.Generator, optional
            Only used if patterns have to be sampled.

        Returns
        -------
        X_NA : torch.Tensor
            Data with NaN values.
        M : torch.Tensor
            Observation mask: 1 = observed, 0 = missing.
        """
        n, d = X.shape

        if patterns is None:
            patterns = self.sample_patterns(d, n, rng)
        patterns = np.asarray(patterns, dtype=int)
        patterns = patterns[~patterns.all(axis=1)]
        n_pats = patterns.shape[0]

        if prop is None:
            prop = n_pats / (n_pats + 1)

        if n_pats > 0:
            result = mice.ampute(
                self.to_r(X),
                patterns=self.to_r(patterns),
                prop=prop,              # freq defaults to equal across patterns
                mech="MAR",
                bycases=True,
                type=self.to_r(miss_type),
                weights=self.to_r(weights)
            )
            amp = self.to_python(result.rx2("amp"))
            amp = amp.to_numpy() if hasattr(amp, "to_numpy") else np.asarray(amp)
            X_NA = torch.tensor(amp, device=X.device, dtype=X.dtype)   # (n, d)

        M = ~torch.isnan(X_NA)

        return X_NA, M