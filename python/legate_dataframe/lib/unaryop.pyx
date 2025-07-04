# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string

from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_data_type

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/unaryop.hpp>" nogil:
    cpp_LogicalColumn cpp_unary_operation "legate::dataframe::unary_operation"(
        const cpp_LogicalColumn& col, string op
    ) except +

    cpp_LogicalColumn cpp_cast "legate::dataframe::cast"(
        const cpp_LogicalColumn& col, data_type dtype
    ) except +


@_track_provenance
def unary_operation(LogicalColumn col,  op: str) -> LogicalColumn:
    """Performs unary operation on all values in column

    Note: For `decimal32` and `decimal64`, only `abs`, `ceil` and `floor` are supported.

    Parameters
    ----------
    col
        Logical column as input
    op
        Operation to perform, see arrow compute functions.

    Returns
    -------
        Logical column of same size as `col` containing result of the operation.
    """
    return LogicalColumn.from_handle(
        cpp_unary_operation(col._handle, op.encode('utf-8'))
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
