# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.scalar cimport cpp_scalar_col_from_python

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/copying.hpp>" nogil:
    cpp_LogicalColumn cpp_copy_if_else "copy_if_else"(
        const cpp_LogicalColumn& lhs,
        const cpp_LogicalColumn& rhs,
        const cpp_LogicalColumn& cond,
    ) except +

    cpp_LogicalColumn cpp_concatenate "concatenate"(
        const vector[cpp_LogicalColumn]& columns,
    ) except +


@_track_provenance
def copy_if_else(
    LogicalColumn cond,
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
) -> LogicalColumn:
    """Performs a ternary if/else operation along the columns.

    The result will contain the values of `lhs[i] if cond[i] else rhs[i]`.
    Both ``lhs`` and ``rhs`` may be scalar columns in which case they are broadcast
    against `cond`. `lhs` and `rhs` must have the same type.

    Parameters
    ----------
    cond
        Boolean column deciding which column each result element is taken from.
    lhs
        The left operand
    lhs
        The right operand

    Returns
    -------
        Output column containing the result of the ternary if/else operation

    Raises
    ------
    ValueError
        If `lhs` and `rhs` do not have the same type or `cond` is not boolean.

    """
    cdef LogicalColumn lhs_col
    cdef LogicalColumn rhs_col
    # If an input is not a column, assume it is scalar:
    if isinstance(lhs, LogicalColumn):
        lhs_col = <LogicalColumn>lhs
    else:
        lhs_col = cpp_scalar_col_from_python(lhs)

    if isinstance(rhs, LogicalColumn):
        rhs_col = <LogicalColumn>rhs
    else:
        rhs_col = cpp_scalar_col_from_python(rhs)

    return LogicalColumn.from_handle(
        cpp_copy_if_else(cond._handle, lhs_col._handle, rhs_col._handle)
    )


@_track_provenance
def concatenate(columns):
    """Concetenate columns into a single long column.

    Creates a new column concatenating all columns.  Must have at
    least one column and all columns must have the same type.

    Parameters
    ----------
    columns
        Iterable of logical columns.

    Returns
    -------
        Output column with as many rows as all input columns combined.
    """
    cdef vector[cpp_LogicalColumn] cpp_cols
    for column in columns:
        if not isinstance(column, LogicalColumn):
            raise TypeError(
                f"columns must be a sequence of LogicalColumn, got "
                f"'{type(column).__name__}'")
        cpp_cols.push_back((<LogicalColumn>column)._handle)

    return LogicalColumn.from_handle(cpp_concatenate(cpp_cols))
