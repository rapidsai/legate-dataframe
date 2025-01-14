# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp cimport bool as cpp_bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from typing import Iterable

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/sort.hpp>" nogil:
    cpp_LogicalTable cpp_sort "legate::dataframe::sort"(
        const cpp_LogicalTable& tbl,
        const vector[string]& keys,
        cpp_bool stable,
    ) except +


@_track_provenance
def sort(
    LogicalTable tbl,
    keys: Iterable[str],
    *,
    stable: bool = False,
):
    """Perform a sort of the table based on the given columns.

    Parameters
    ----------
    tbl
        The table to sort
    keys
        The column names to sort by.
    stable
        Whether to perform a stable sort (default ``False``).

    Returns
    -------
        A new sorted table.

    """
    cdef vector[string] keys_vector
    for k in keys:
        keys_vector.push_back(k.encode('UTF-8'))

    if keys_vector.size() == 0:
        raise ValueError("sort keys must contain at least one column.")

    return LogicalTable.from_handle(cpp_sort(tbl._handle, keys_vector, stable))
