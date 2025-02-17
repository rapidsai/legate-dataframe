# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from pylibcudf.unary cimport unary_operator as cpp_unary_operator

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn

from pylibcudf.unary import UnaryOperator as unary_operator  # no-cython-lint

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/unaryop.hpp>" nogil:
    cpp_LogicalColumn cpp_unary_operation "legate::dataframe::unary_operation"(
        const cpp_LogicalColumn& col, cpp_unary_operator op
    ) except +


@_track_provenance
def unary_operation(LogicalColumn col, cpp_unary_operator op) -> LogicalColumn:
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
