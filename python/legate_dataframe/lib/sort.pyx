# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp cimport bool as cpp_bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/sort.hpp>" nogil:
    cpp_LogicalTable cpp_sort "legate::dataframe::sort"(
        const cpp_LogicalTable& tbl,
        const vector[string]& keys,
        const vector[cpp_bool]& sort_ascending,
        cpp_bool nulls_at_end,
        cpp_bool stable,
    ) except +


@_track_provenance
def sort(
    LogicalTable tbl,
    list keys,
    *,
    list sort_ascending = None,
    nulls_at_end = True,
    stable = False,
):
    """Perform a sort of the table based on the given columns.
    The GPU and CPU backends may not sort NaN values exactly the same way
    (e.g. according to null_precendence or by treating them as large
    floating point numbers) - it is recommended to instead use nulls
    instead of NaNs to get a consistent behaviour between CPU/GPU launches.


    Parameters
    ----------
    tbl
        The table to sort
    keys
        The column names to sort by.
    sort_ascending
        A list of boolean values for each key denoting whether to sort in
        ascending (True) or descending (False) order. Defaults to all ascending.
    nulls_at_end
        Whether NULL values should be placed at the end (True) or beginning (False)
        of the sorted result. Defaults to True (nulls at end).
    stable
        Whether to perform a stable sort (default ``False``).  Stable sort currently
        uses a less efficient merge and may not perform as well as it should.

    Returns
    -------
        A new sorted table.

    """
    cdef vector[string] keys_vector
    cdef vector[cpp_bool] c_sort_ascending
    cdef cpp_bool c_nulls_at_end = nulls_at_end

    if sort_ascending is None:
        c_sort_ascending = [True] * len(keys)
    else:
        c_sort_ascending = sort_ascending

    for k in keys:
        keys_vector.push_back(k.encode('UTF-8'))

    return LogicalTable.from_handle(
        cpp_sort(tbl._handle, keys_vector, c_sort_ascending, c_nulls_at_end, stable))
