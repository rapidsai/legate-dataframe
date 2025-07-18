# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libc.stdint import int64_t
from libcpp cimport bool as cpp_bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibcudf.types cimport null_order, order

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from pylibcudf.types import NullOrder, Order

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/sort.hpp>" nogil:
    cpp_LogicalTable cpp_sort "legate::dataframe::sort"(
        const cpp_LogicalTable& tbl,
        const vector[string]& keys,
        const vector[order]& c_order,
        const vector[null_order]& c_null_precedence,
        cpp_bool stable,
        optional[int64_t] limit,
    ) except +


@_track_provenance
def sort(
    LogicalTable tbl,
    list keys,
    *,
    list column_order = None,
    list null_precedence = None,
    stable = False,
    limit = None
):
    """Perform a sort of the table based on the given columns.

    Parameters
    ----------
    tbl
        The table to sort
    keys
        The column names to sort by.
    column_order
        An ``Order.ASCENDING`` or ``Order.DESCENDING`` for each key denoting the
        final order for that column.  Defaults to all ascending.
    null_precedence
        A ``NullOrder.BEFORE`` or ``NullOrder.AFTER`` for each key denoting if NULL
        values are considered considered smaller (before) or larger (after) any
        value.  I.e. by default nulls are sorted "after" meaning they come
        last after an ascending sort and first after a descending sort.
    stable
        Whether to perform a stable sort (default ``False``).  Stable sort currently
        uses a less efficient merge and may not perform as well as it should.
    limit
        Maximum number of rows to return. If positive, returns the first, if negative
        the last. (In a distributed setting, this reduces the amount of data exchanged.)

    Returns
    -------
        A new sorted table.

    """
    cdef vector[string] keys_vector
    cdef vector[order] c_orders
    cdef vector[null_order] c_null_precedence
    cdef optional[int64_t] cpp_limit

    if column_order is None:
        c_orders = [Order.ASCENDING] * len(keys)
    else:
        c_orders = column_order

    if null_precedence is None:
        c_null_precedence = [NullOrder.AFTER] * len(keys)
    else:
        c_null_precedence = null_precedence

    if limit is not None:
        cpp_limit = <int64_t>limit

    for k in keys:
        keys_vector.push_back(k.encode('UTF-8'))

    return LogicalTable.from_handle(cpp_sort(
        tbl._handle, keys_vector, c_orders, c_null_precedence, stable, cpp_limit))
