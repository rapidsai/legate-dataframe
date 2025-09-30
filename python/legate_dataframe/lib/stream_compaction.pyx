# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool as cpp_bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/stream_compaction.hpp>" nogil:
    cpp_LogicalTable cpp_apply_boolean_mask "legate::dataframe::apply_boolean_mask"(
        const cpp_LogicalTable& tbl,
        const cpp_LogicalColumn& boolean_mask,
    ) except +
    cpp_LogicalTable cpp_distinct "legate::dataframe::distinct"(
        const cpp_LogicalTable& tbl,
        const vector[string] keys,
        cpp_bool high_cardinality,
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


@_track_provenance
def distinct(
    LogicalTable tbl,
    keys=None,
    *,
    high_cardinality=False,
):
    """Filter a table busing a boolean mask.

    Select all rows from the table where the boolean mask column is true
    (non-null and not false).  The operation is stable.

    Parameters
    ----------
    tbl
        The table to filter.
    keys
        Keys that must be distinct.  If not given, use all columns.
    high_cardinality : bool
        If set to ``True`` assume the result has high cardinality.
        Otherwise (current default), assumes the result has low cardinality.
        If the result has low cardinality (many duplicate rows) it is useful
        to perform a local distinct to reduce communication. In the worst case
        (all unique) this doubles the non-communication part of the work.

    Returns
    -------
        The ``LogicalTable`` containing only distinct rows.
    """
    cdef vector[string] cpp_keys

    if keys is None:
        keys = tbl.get_column_names()  # all names
    for key in keys:
        cpp_keys.push_back(key.encode('UTF-8'))

    return LogicalTable.from_handle(
        cpp_distinct(tbl._handle, cpp_keys, high_cardinality))
