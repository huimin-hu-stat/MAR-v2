from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter


class Converter:
    """
    Convert between Python and R objects while preserving array dimensions.

    Supported Python inputs:
        - NumPy arrays and scalars
        - pandas DataFrames and Series
        - PyTorch tensors (if installed)
        - Python scalars, lists, tuples, and dictionaries

    R conversion:
        - 1D numeric array -> R vector
        - 2D numeric array -> R matrix
        - ND numeric array -> R array
        - pandas DataFrame -> R data.frame

    Notes:
        - Tensor dimensions are preserved.
        - GPU tensors are moved to CPU before conversion.
        - Python dictionaries become R lists, not data frames.
        - NaN is preserved as a floating-point NaN; R's is.na()
          treats both NA and NaN as missing.
    """

    def __init__(self):
        self._converter = (
            default_converter
            + numpy2ri.converter
            + pandas2ri.converter
        )

    @staticmethod
    def _torch_to_numpy(x: Any) -> np.ndarray:
        """Convert a PyTorch tensor to a NumPy array safely."""
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()

            # NumPy cannot directly represent every PyTorch dtype.
            if x.dtype == torch.bfloat16:
                x = x.float()

            if x.is_complex():
                return x.numpy()

            return x.numpy()

        raise TypeError("Expected a PyTorch tensor.")

    def to_r(self, x: Any) -> Any:
        """Convert a Python object to its corresponding R representation."""

        # 0. None -> R NULL
        if x is None:
            return ro.NULL
        
        # 1. PyTorch tensor
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = self._torch_to_numpy(x)
        except ImportError:
            pass

        # 2. NumPy arrays: explicitly preserve dimensions
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "O":
                raise TypeError(
                    "Object-dtype NumPy arrays are ambiguous. "
                    "Convert the array to a supported numeric, "
                    "string, or boolean dtype first."
                )

            with localconverter(self._converter):
                return ro.conversion.get_conversion().py2rpy(x)

        # 3. NumPy scalar
        if isinstance(x, np.generic):
            x = x.item()

        # 4. pandas DataFrame or Series
        if isinstance(x, (pd.DataFrame, pd.Series)):
            with localconverter(self._converter):
                return ro.conversion.get_conversion().py2rpy(x)

        # 5. Dictionary -> R list
        if isinstance(x, dict):
            with localconverter(self._converter):
                values = {
                    str(k): self.to_r(v)
                    for k, v in x.items()
                }
                return ro.ListVector(values)

        # 6. List or tuple -> R list
        if isinstance(x, (list, tuple)):
            with localconverter(self._converter):
                values = [self.to_r(v) for v in x]
                return ro.ListVector(
                    {str(i + 1): v for i, v in enumerate(values)}
                )

        # 7. Python scalars and already-convertible objects
        with localconverter(self._converter):
            return ro.conversion.get_conversion().py2rpy(x)

    def to_python(self, x: Any) -> Any:
        """
        Convert an R object to Python.

        R matrices and arrays are converted to NumPy arrays with
        their dimensions preserved; R data frames become pandas
        DataFrames when supported by the installed rpy2 version.
        """
        with localconverter(self._converter):
            result = ro.conversion.get_conversion().rpy2py(x)

        return result

    def to_tensor(
        self,
        x: Any,
        *,
        dtype=None,
        device=None,
    ):
        """Convert an array-like object to a PyTorch tensor."""
        import torch

        if isinstance(x, torch.Tensor):
            result = x
        else:
            if isinstance(x, (ro.vectors.Array, ro.vectors.Vector)):
                x = self.to_python(x)

            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = x.to_numpy()

            result = torch.as_tensor(np.asarray(x))

        if dtype is not None:
            result = result.to(dtype=dtype)

        if device is not None:
            result = result.to(device=device)

        return result

    @staticmethod
    def shape(x: Any) -> tuple:
        """Return an object's dimensions where available."""
        if hasattr(x, "shape"):
            return tuple(x.shape)

        if hasattr(x, "dim"):
            dims = x.dim
            if dims is not ro.NULL:
                return tuple(int(d) for d in dims)

        return ()

    @staticmethod
    def is_r_array(x: Any) -> bool:
        """Check whether an R object has a dim attribute."""
        if not hasattr(x, "dim"):
            return False

        try:
            return x.dim is not ro.NULL
        except (AttributeError, TypeError):
            return False