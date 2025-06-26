# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.scalar cimport cpp_scalar_col_from_python

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/replace.hpp>" nogil:
    cpp_LogicalColumn cpp_replace_nulls "replace_nulls"(
        const cpp_LogicalColumn& col,
        const cpp_LogicalColumn& value,
    ) except +


@_track_provenance
def replace_nulls(
    LogicalColumn col,
    replacement: ScalarLike,
) -> LogicalColumn:
    """Return a new column with NULL entries replaced by value.

    Parameters
    ----------
    lhs
        Operand column
    replacement
        Value to replace NULLs with (currently limited to scalars).

    Returns
    -------
        Output column of `output_type` type without NULL entries.

    Raises
    ------
    ValueError: if the value is not of the correct scalar type.

    """
    cdef LogicalColumn repl_col = cpp_scalar_col_from_python(replacement)
    return LogicalColumn.from_handle(
        cpp_replace_nulls(col._handle, repl_col._handle)
    )
