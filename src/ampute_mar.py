import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

mice = importr("mice")

class AmputeMAR:

    def _to_r_dataframe(self, X):
        return ro.DataFrame({
            f"X{j + 1}": ro.FloatVector(X[:, j])
            for j in range(X.shape[1])
        })

    def _to_r_matrix(self, X):
        return ro.r.matrix(
            ro.IntVector(X.T.flatten().tolist()),
            nrow=X.shape[0],
            ncol=X.shape[1],
        )

    @staticmethod
    def sample_patterns(d, n, rng=None):
        """Sample incomplete missingness patterns once (1 = observed, 0 = missing).
        Last column is always observed; all-ones rows are excluded."""
        rng = np.random.default_rng() if rng is None else rng
        while True:
            npattern = int(rng.integers(3, max(round(n / 100), 3) + 1))
            patterns = rng.integers(0, 2, size=(npattern, d - 1))
            patterns = np.hstack([patterns, np.ones((npattern, 1), dtype=int)])
            incomplete = patterns[~patterns.all(axis=1)]
            if incomplete.shape[0] > 0:
                return incomplete

    def ampute(self, X, patterns=None, prop=None, rng=None):
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
        device = X.device
        X_np = X.detach().cpu().numpy().astype(float)

        if patterns is None:
            patterns = self.sample_patterns(d, n, rng)
        patterns = np.asarray(patterns, dtype=int)
        patterns = patterns[~patterns.all(axis=1)]
        n_pats = patterns.shape[0]

        if prop is None:
            prop = n_pats / (n_pats + 1)

        X_NA_np = X_np.copy()

        if n_pats > 0:
            result = mice.ampute(
                self._to_r_dataframe(X_np),
                patterns=self._to_r_matrix(patterns),
                prop=prop,              # freq defaults to equal across patterns
                mech="MAR",
                bycases=True,
            )
            X_NA_np = np.asarray(result.rx2("amp")).T   # (n, d)

        M_np = (~np.isnan(X_NA_np)).astype(np.float32)

        return (
            torch.tensor(X_NA_np, dtype=X.dtype, device=device),
            torch.tensor(M_np, dtype=X.dtype, device=device),
        )