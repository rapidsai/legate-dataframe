# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference

from legate_dataframe.lib.core.table cimport LogicalColumn, cpp_LogicalColumn

from legate_dataframe.utils import _track_provenance

from pylibcudf.aggregation cimport Aggregation
from pylibcudf.libcudf.aggregation cimport reduce_aggregation


cdef extern from "<legate_dataframe/reduction.hpp>" nogil:
    cpp_LogicalColumn cpp_reduce "legate::dataframe::reduce"(
        cpp_LogicalColumn& col, const reduce_aggregation& agg
    ) except +


@_track_provenance
def reduce(LogicalColumn column, Aggregation agg):
    """Apply a reduction along a column.

    Parameters
    ----------
    column
        The column to reduce.
    agg
        The `pylibcudf.aggregation.Aggregation` to apply.
    """
    cdef const reduce_aggregation *cpp_agg = agg.view_underlying_as_reduce()

    return LogicalColumn.from_handle(cpp_reduce(column._handle, dereference(cpp_agg)))
