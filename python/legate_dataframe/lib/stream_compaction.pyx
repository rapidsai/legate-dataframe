# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/stream_compaction.hpp>" nogil:
    cpp_LogicalTable cpp_apply_boolean_mask "legate::dataframe::apply_boolean_mask"(
        const cpp_LogicalTable& tbl,
        const cpp_LogicalColumn& boolean_mask,
    ) except +


@_track_provenance
def apply_boolean_mask(
    LogicalTable tbl,
    LogicalColumn boolean_mask,
):
    """Filter a table busing a boolean mask.

    Select all rows from the table where the boolean mask column is true
    (non-null and not false).  The operation is stable.

    Parameters
    ----------
    tbl
        The table to filter.
    boolean_mask
        The boolean mask to apply.

    Returns
    -------
        The ``LogicalTable`` containing only the rows where the boolean_mask was true.

    """
    return LogicalTable.from_handle(
        cpp_apply_boolean_mask(tbl._handle, boolean_mask._handle))
