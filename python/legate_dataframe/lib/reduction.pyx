# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference

from pylibcudf.aggregation cimport Aggregation
from pylibcudf.libcudf.aggregation cimport reduce_aggregation
from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.data_type cimport as_data_type
from legate_dataframe.lib.core.scalar cimport cpp_scalar_col_from_python
from legate_dataframe.lib.core.table cimport LogicalColumn, cpp_LogicalColumn

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/reduction.hpp>" nogil:
    cpp_LogicalColumn cpp_reduce "legate::dataframe::reduce"(
        cpp_LogicalColumn& col, const reduce_aggregation& agg, data_type output_type,
    ) except +
    cpp_LogicalColumn cpp_reduce "legate::dataframe::reduce"(
        cpp_LogicalColumn& col, const reduce_aggregation& agg, data_type output_type,
        cpp_LogicalColumn& scalar  # actually an optional[reference_type(LogicalColumn)]
    ) except +


@_track_provenance
def reduce(
    LogicalColumn col, Aggregation agg, output_type, *, initial=None
):
    """Apply a reduction along a column.

    Parameters
    ----------
    col
        The column to reduce.
    agg
        The ``pylibcudf.aggregation.Aggregation`` to apply, see
        :external:py:mod:`pylibcudf.aggregation`.
        For example, pass ``pylibcudf.aggregation.sum()`` to sum the column.
    output_type
        The result dtype, must be specified.
    initial
        Scalar column containing an initial value for the reduction.
    """
    cdef const reduce_aggregation *cpp_agg = agg.view_underlying_as_reduce()
    cdef data_type otype = as_data_type(output_type)
    cdef LogicalColumn initial_col

    if initial is None:
        return LogicalColumn.from_handle(
            cpp_reduce(col._handle, dereference(cpp_agg), otype)
        )
    else:
        initial_col = cpp_scalar_col_from_python(initial)
        return LogicalColumn.from_handle(
            cpp_reduce(col._handle, dereference(cpp_agg), otype, initial_col._handle)
        )
