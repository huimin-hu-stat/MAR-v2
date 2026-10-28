
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

    Python -> R
    -----------
    - NumPy arrays and scalars
    - pandas DataFrames and Series
    - PyTorch tensors (if installed)
    - Python scalars, lists, tuples, and dictionaries

    R -> Python
    -----------
    - Named R lists -> Python dictionaries
    - Unnamed R lists -> Python lists
    - Nested R lists -> recursively converted containers
    - R matrices and arrays -> NumPy arrays, preserving dimensions
    - R data frames -> pandas DataFrames when supported

    Notes
    -----
    - GPU tensors are moved to CPU before conversion.
    - Python dictionaries become R lists, not data frames.
    - NumPy arrays with object dtype are rejected.
    - R objects should not generally be passed between Joblib
      worker processes. Create/access and convert them in the worker.
    """

    def __init__(self):
        self._converter = (
            default_converter
            + numpy2ri.converter
            + pandas2ri.converter
        )

    # ---------------------------------------------------------
    # PyTorch -> NumPy
    # ---------------------------------------------------------

    @staticmethod
    def _torch_to_numpy(x: Any) -> np.ndarray:
        """Convert a PyTorch tensor to a NumPy array safely."""
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a PyTorch tensor.")

        x = x.detach().cpu()

        # NumPy does not support every PyTorch dtype.
        if x.dtype == torch.bfloat16:
            x = x.float()

        return x.numpy()

    # ---------------------------------------------------------
    # Python -> R
    # ---------------------------------------------------------

    def to_r(self, x: Any) -> Any:
        """Convert a Python object to its R representation."""

        # None -> R NULL
        if x is None:
            return ro.NULL

        # PyTorch tensor -> NumPy array
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = self._torch_to_numpy(x)

        except ImportError:
            pass

        # NumPy array: preserve dimensions
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "O":
                raise TypeError(
                    "Object-dtype NumPy arrays are ambiguous. "
                    "Convert to a supported dtype first."
                )

            with localconverter(self._converter):
                return ro.conversion.get_conversion().py2rpy(x)

        # NumPy scalar -> Python scalar
        if isinstance(x, np.generic):
            x = x.item()

        # pandas DataFrame / Series
        if isinstance(x, (pd.DataFrame, pd.Series)):
            with localconverter(self._converter):
                return ro.conversion.get_conversion().py2rpy(x)

        # Dictionary -> named R list
        if isinstance(x, dict):
            values = {
                str(k): self.to_r(v)
                for k, v in x.items()
            }

            return ro.ListVector(values)

        # List / tuple -> R list
        if isinstance(x, (list, tuple)):
            values = [
                self.to_r(v)
                for v in x
            ]

            return ro.ListVector(
                {
                    str(i + 1): value
                    for i, value in enumerate(values)
                }
            )

        # Python scalars and other supported objects
        with localconverter(self._converter):
            return ro.conversion.get_conversion().py2rpy(x)

    # ---------------------------------------------------------
    # R -> Python
    # ---------------------------------------------------------

    def to_python(self, x: Any) -> Any:
        """
        Convert an R object to Python.

        Named R lists become dictionaries; unnamed R lists become
        lists. Nested lists are converted recursively.
        """

        # R NULL -> None
        if x is ro.NULL:
            return None

        # Handle R lists explicitly before generic conversion.
        if isinstance(x, ro.vectors.ListVector):
            names = x.names

            # No names: return a Python list.
            if names is ro.NULL:
                return [
                    self.to_python(x[i])
                    for i in range(len(x))
                ]

            names = list(names)

            # All elements have nonempty names: return a dictionary.
            if all(name not in (None, "") for name in names):
                return {
                    str(name): self.to_python(x[i])
                    for i, name in enumerate(names)
                }

            # Partially named list: preserve all elements and positions.
            # Named elements become dictionary entries; unnamed elements
            # remain in the list so that no values are lost.
            return [
                (
                    {str(name): self.to_python(x[i])}
                    if name not in (None, "")
                    else self.to_python(x[i])
                )
                for i, name in enumerate(names)
            ]

        # Atomic vectors, matrices, arrays, data frames, etc.
        with localconverter(self._converter):
            return ro.conversion.get_conversion().rpy2py(x)

    # ---------------------------------------------------------
    # R -> PyTorch
    # ---------------------------------------------------------

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
            # Convert R vectors, matrices, arrays, and lists first.
            if isinstance(
                x,
                (
                    ro.vectors.Array,
                    ro.vectors.ListVector,
                ),
            ):
                x = self.to_python(x)

            if isinstance(x, (pd.DataFrame, pd.Series)):
                x = x.to_numpy()

            result = torch.as_tensor(np.asarray(x))

        if dtype is not None:
            result = result.to(dtype=dtype)

        if device is not None:
            result = result.to(device=device)

        return result

    # ---------------------------------------------------------
    # Shape utilities
    # ---------------------------------------------------------

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


    def __getstate__(self):
        """Exclude rpy2 converter machinery from serialized state."""
        return {}


    def __setstate__(self, state):
        """Recreate conversion machinery in the receiving process."""
        self._converter = (
            default_converter
            + numpy2ri.converter
            + pandas2ri.converter
    )