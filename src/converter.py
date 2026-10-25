
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri, pandas2ri, vectors


# Optional PyTorch support
try:
    import torch
except ImportError:
    torch = None

# Optional TensorFlow support
try:
    import tensorflow as tf
except ImportError:
    tf = None


class Converter:
    """
    Converter between R and Python.

    Supported Python objects
    ------------------------
    - None, bool, int, float, complex, str, bytes
    - list, tuple, dict, set
    - numpy scalars and arrays
    - pandas Series, DataFrame, Index
    - PyTorch tensors (optional)
    - TensorFlow tensors (optional)

    Supported R objects
    -------------------
    - NULL
    - Logical, integer, numeric, complex, character vectors
    - Lists and named lists
    - Matrices and multidimensional arrays
    - Data frames
    - Factors
    - Date and POSIXct objects, subject to rpy2 conversion support

    Notes
    -----
    - Requires a working R installation and rpy2.
    - Tensor conversion transfers data to CPU and does not preserve
      PyTorch autograd graphs or TensorFlow gradient tapes.
    - Not every R attribute or Python metadata field survives a
      round trip.
    """

    def __init__(self):
        self.converter = (
            default_converter
            + numpy2ri.converter
            + pandas2ri.converter
        )

    # =========================================================
    # Core public interface
    # =========================================================

    def to_r(self, obj: Any):
        """Convert a supported Python object to an R object."""

        with conversion.localconverter(self.converter):
            return self._to_r(obj)

    def to_python(self, obj: Any):
        """Convert an R object to its Python representation."""

        with conversion.localconverter(self.converter):
            return self._to_python(obj)

    # =========================================================
    # Python -> R
    # =========================================================

    def _to_r(self, obj: Any):
        """Recursively convert Python objects to R objects."""

        # None -> R NULL
        if obj is None:
            return ro.NULL

        # Already an R object
        if isinstance(obj, ro.RObject):
            return obj

        # PyTorch tensor -> NumPy -> R
        if torch is not None and isinstance(obj, torch.Tensor):
            arr = self.tensor_to_numpy(obj)
            return self._to_r(arr)

        # TensorFlow tensor -> NumPy -> R
        if tf is not None and tf.is_tensor(obj):
            arr = self.tensor_to_numpy(obj)
            return self._to_r(arr)

        # pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return conversion.get_conversion().py2rpy(obj)

        # pandas Series
        if isinstance(obj, pd.Series):
            return conversion.get_conversion().py2rpy(obj)

        # pandas Index
        if isinstance(obj, pd.Index):
            return self._to_r(obj.to_numpy())

        # NumPy array
        if isinstance(obj, np.ndarray):
            return conversion.get_conversion().py2rpy(obj)

        # NumPy scalar
        if isinstance(obj, np.generic):
            obj = obj.item()

        # Python scalar types
        if isinstance(obj, bool):
            return vectors.BoolVector([obj])

        if isinstance(obj, int):
            # R integer vectors are signed 32-bit integers.
            if -(2**31) < obj < 2**31:
                return vectors.IntVector([obj])
            return vectors.FloatVector([float(obj)])

        if isinstance(obj, float):
            return vectors.FloatVector([obj])

        if isinstance(obj, complex):
            return vectors.ComplexVector([obj])

        if isinstance(obj, str):
            return vectors.StrVector([obj])

        if isinstance(obj, bytes):
            return vectors.ByteVector([obj])

        # Python dictionary -> named R list
        if isinstance(obj, Mapping):
            values = [self._to_r(v) for v in obj.values()]
            names = [str(k) for k in obj.keys()]

            return vectors.ListVector(
                dict(zip(names, values))
            )

        # Python list / tuple
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return vectors.ListVector([])

            # Preserve nested structures as R lists.
            if any(
                isinstance(x, (Mapping, list, tuple, pd.DataFrame))
                or (
                    torch is not None
                    and isinstance(x, torch.Tensor)
                )
                or (
                    tf is not None
                    and tf.is_tensor(x)
                )
                for x in obj
            ):
                return vectors.ListVector(
                    [self._to_r(x) for x in obj]
                )

            # Try homogeneous NumPy representation.
            try:
                arr = np.asarray(obj)

                if arr.dtype.kind in "biufc":
                    return conversion.get_conversion().py2rpy(arr)

                if arr.dtype.kind in "U":
                    return vectors.StrVector(arr.tolist())

            except (TypeError, ValueError):
                pass

            # Character vectors
            if all(isinstance(x, str) for x in obj):
                return vectors.StrVector(obj)

            # Heterogeneous objects -> R list
            return vectors.ListVector(
                [self._to_r(x) for x in obj]
            )

        # Sets have no guaranteed order.
        if isinstance(obj, set):
            return self._to_r(list(obj))

        raise TypeError(
            f"Unsupported Python type: {type(obj).__name__}"
        )

    # =========================================================
    # R -> Python
    # =========================================================

    def _to_python(self, obj: Any):
        """Recursively convert R objects to Python objects."""

        if obj is ro.NULL:
            return None

        if not isinstance(obj, ro.RObject):
            return obj

        # Named or unnamed R lists
        if isinstance(obj, vectors.ListVector):
            values = [self._to_python(x) for x in obj]

            names_obj = obj.names

            if names_obj is not ro.NULL:
                names = list(names_obj)

                if (
                    len(names) == len(values)
                    and all(name != "" for name in names)
                ):
                    # A dict cannot represent duplicate keys without loss.
                    if len(set(names)) == len(names):
                        return dict(zip(names, values))

            return values

        # Delegate DataFrames, vectors, matrices, factors, and dates
        # to the registered rpy2 converters.
        result = conversion.get_conversion().rpy2py(obj)

        # Normalize NumPy scalar types.
        if isinstance(result, np.generic):
            return result.item()

        return result

    # =========================================================
    # Tensor conversion helpers
    # =========================================================

    def tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert a PyTorch or TensorFlow tensor to a NumPy array.

        PyTorch tensors are detached and moved to CPU first.
        """

        if torch is not None and isinstance(tensor, torch.Tensor):
            # Handle bfloat16, which NumPy may not support directly.
            tensor = tensor.detach().cpu()

            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            return tensor.numpy()

        if tf is not None and tf.is_tensor(tensor):
            return tensor.numpy()

        raise TypeError(
            f"Expected a PyTorch or TensorFlow tensor; "
            f"received {type(tensor).__name__}."
        )

    def tensor_to_r(self, tensor: Any):
        """
        Convert a tensor to an R vector, matrix, or array.

        0D tensor -> length-one R vector
        1D tensor -> R vector
        2D tensor -> R matrix
        ND tensor -> R array
        """

        return self.to_r(self.tensor_to_numpy(tensor))

    def r_to_tensor(
        self,
        obj: Any,
        framework: str = "torch",
        dtype: Any = None,
        device: Any = None,
    ):
        """
        Convert an R vector, matrix, or array to a tensor.

        Parameters
        ----------
        obj:
            An R object.
        framework:
            "torch" or "tensorflow".
        dtype:
            Optional target framework dtype.
        device:
            Optional PyTorch device, e.g. "cpu" or "cuda".
        """

        arr = np.asarray(self.to_python(obj))

        if framework == "torch":
            if torch is None:
                raise ImportError(
                    "PyTorch is not installed. Run pip install torch."
                )

            tensor = torch.as_tensor(arr, dtype=dtype)

            if device is not None:
                tensor = tensor.to(device)

            return tensor

        if framework == "tensorflow":
            if tf is None:
                raise ImportError(
                    "TensorFlow is not installed. "
                    "Run pip install tensorflow."
                )

            return tf.convert_to_tensor(arr, dtype=dtype)

        raise ValueError(
            "framework must be 'torch' or 'tensorflow'."
        )

    # =========================================================
    # Explicit Python -> R methods
    # =========================================================

    def dict_to_r_list(self, obj: dict):
        """Python dict -> named R list."""
        return self.to_r(obj)

    def list_to_r(self, obj: list):
        """Python list -> R vector or list."""
        return self.to_r(obj)

    def tuple_to_r(self, obj: tuple):
        """Python tuple -> R vector or list."""
        return self.to_r(obj)

    def array_to_r(self, obj: np.ndarray):
        """NumPy array -> R vector, matrix, or array."""
        return self.to_r(obj)

    def series_to_r(self, obj: pd.Series):
        """pandas Series -> R vector."""
        return self.to_r(obj)

    def dataframe_to_r(self, obj: pd.DataFrame):
        """pandas DataFrame -> R data.frame."""
        return self.to_r(obj)

    # =========================================================
    # Explicit R -> Python methods
    # =========================================================

    def r_list_to_dict(self, obj):
        """Convert a named R list to a Python dictionary."""
        result = self.to_python(obj)

        if not isinstance(result, dict):
            raise TypeError("Expected a named R list.")

        return result

    def r_vector_to_numpy(self, obj):
        """R vector -> NumPy array."""
        return np.asarray(self.to_python(obj))

    def r_matrix_to_numpy(self, obj):
        """R matrix -> two-dimensional NumPy array."""
        result = np.asarray(self.to_python(obj))

        if result.ndim != 2:
            raise TypeError("Expected a two-dimensional R matrix.")

        return result

    def r_array_to_numpy(self, obj):
        """R array -> NumPy array, preserving dimensions."""
        return np.asarray(self.to_python(obj))

    def r_dataframe_to_pandas(self, obj):
        """R data.frame -> pandas DataFrame."""
        result = self.to_python(obj)

        if not isinstance(result, pd.DataFrame):
            raise TypeError("Expected an R data.frame.")

        return result

    def r_factor_to_pandas(self, obj):
        """R factor -> pandas categorical Series."""
        result = self.to_python(obj)

        if isinstance(result, pd.Series):
            return result.astype("category")

        return pd.Series(result, dtype="category")

    # =========================================================
    # R evaluation helpers
    # =========================================================

    def eval_r(self, expression: str):
        """
        Evaluate an R expression and convert the result to Python.

        Example
        -------
        converter.eval_r("data.frame(x = 1:3)")
        """
        result = ro.r(expression)
        return self.to_python(result)

    def exec_r(self, expression: str):
        """
        Execute R code and return the raw R result.

        Use this when you want to retain the R object.
        """
        return ro.r(expression)
