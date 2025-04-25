# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport int32_t

from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_data_type
from legate_dataframe.utils import _track_provenance


cdef extern from "<cudf/unary.hpp>" namespace "cudf":
    cpdef enum class unary_operator(int32_t):
        """Enum of unary operators, see :external:cpp:enum:`unary_operator`.
        """
        SIN,         # Trigonometric sine
        COS,         # Trigonometric cosine
        TAN,         # Trigonometric tangent
        ARCSIN,      # Trigonometric sine inverse
        ARCCOS,      # Trigonometric cosine inverse
        ARCTAN,      # Trigonometric tangent inverse
        SINH,        # Hyperbolic sine
        COSH,        # Hyperbolic cosine
        TANH,        # Hyperbolic tangent
        ARCSINH,     # Hyperbolic sine inverse
        ARCCOSH,     # Hyperbolic cosine inverse
        ARCTANH,     # Hyperbolic tangent inverse
        EXP,         # Exponential (base e, Euler number)
        LOG,         # Natural Logarithm (base e)
        SQRT,        # Square-root (x^0.5)
        CBRT,        # Cube-root (x^(1.0/3))
        CEIL,        # Smallest integer value not less than arg
        FLOOR,       # largest integer value not greater than arg
        ABS,         # Absolute value
        RINT,        # Rounds the floating-point argument arg to an integer value
        BIT_INVERT,  # Bitwise Not (~)
        NOT,         # Logical Not (!)


cdef extern from "<legate_dataframe/unaryop.hpp>" nogil:
    cpp_LogicalColumn cpp_unary_operation "legate::dataframe::unary_operation"(
        const cpp_LogicalColumn& col, unary_operator op
    ) except +

    cpp_LogicalColumn cpp_cast "legate::dataframe::cast"(
        const cpp_LogicalColumn& col, data_type dtype
    ) except +


@_track_provenance
def unary_operation(LogicalColumn col, unary_operator op) -> LogicalColumn:
    """Performs unary operation on all values in column

    Note: For `decimal32` and `decimal64`, only `ABS`, `CEIL` and `FLOOR` are supported.

    Parameters
    ----------
    col
        Logical column as input
    op
        Operation to perform, see `unary_operator`.

    Returns
    -------
        Logical column of same size as `col` containing result of the operation.
    """
    return LogicalColumn.from_handle(
        cpp_unary_operation(col._handle, op)
    )


@_track_provenance
def cast(LogicalColumn col, dtype) -> LogicalColumn:
    """Cast a logical column to the desired data type.

    Parameters
    ----------
    col
        Logical column as input
    dtype
        The cudf data type of the result.

    Returns
    -------
        Logical column of same size as `col` but with new data type.
    """
    return LogicalColumn.from_handle(
        cpp_cast(col._handle, as_data_type(dtype))
    )
