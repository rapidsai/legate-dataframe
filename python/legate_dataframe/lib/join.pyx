# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libc.stdint cimport int32_t
from libcpp cimport bool as cpp_bool
from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from typing import Iterable, Optional

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/join.hpp>" namespace "legate::dataframe":
    cpdef enum class JoinType(int32_t):
        """Options for the join type (``INNER``, ``LEFT``, ``FULL``, ``SEMI``,
        ``ANTI``).
        """
        INNER,
        LEFT,
        FULL,
        SEMI,
        ANTI

    cpdef enum class BroadcastInput(int32_t):
        """Options for input broadcasting (``AUTO``, ``LEFT``, and ``RIGHT``).
        """
        AUTO,
        LEFT,
        RIGHT

cdef extern from "<legate_dataframe/join.hpp>" nogil:
    cpp_LogicalTable cpp_join "legate::dataframe::join"(
        const cpp_LogicalTable& lhs,
        const cpp_LogicalTable& rhs,
        const vector[string]& lhs_keys,
        const vector[string]& rhs_keys,
        JoinType join_type,
        const vector[string]& lhs_out_columns,
        const vector[string]& rhs_out_columns,
        cpp_bool nulls_equal,
        BroadcastInput broadcast
    ) except +


@_track_provenance
def join(
    LogicalTable lhs,
    LogicalTable rhs,
    *,
    lhs_keys: Iterable[str],
    rhs_keys: Iterable[str],
    JoinType join_type,
    lhs_out_columns: Optional[Iterable[str]] = None,
    rhs_out_columns: Optional[Iterable[str]] = None,
    nulls_equal: bool = True,
    BroadcastInput broadcast = BroadcastInput.AUTO
):
    """Perform an join between the specified tables.

    By default, the returned Table includes the columns from both `lhs` and `rhs`.
    In order to select the desired output columns, please use the `lhs_out_columns`
    and `rhs_out_columns` arguments. This can be useful to avoid duplicate
    key names and columns.

    Parameters
    ----------
    lhs
        The left table
    rhs
        The right table
    lhs_keys
        The column names of the left table to join on
    rhs_keys
        The column names of the right table to join on
    join_type
        The `JoinType` such as ``INNER``, ``LEFT``, ``FULL``
    lhs_out_columns
        Left table column names to include in the result. If None,
        all columns are included. All names in `lhs_out_columns` and `rhs_out_columns`
        must be unique.
    rhs_out_columns
        Right table column names to include in the result. If None,
        all columns are included. All names in `lhs_out_columns` and `rhs_out_columns`
        must be unique.
    nulls_equal
        Controls whether null join-key values should match or not
    broadcast : BroadcastInput
        Can be ``RIGHT`` or ``LEFT`` to indicate that the array is "broadcast"
        to all workers (i.e. copied fully).  This can be much faster,
        as it avoids more complex all-to-all communication.
        Defaults to ``AUTO`` which may do this based on the data size.

    Returns
    -------
        The result of the join, which include the columns specified in `lhs_out_columns`
        and `rhs_out_columns` (in that order).

    Raises
    ------
    ValueError
        If number of elements in `lhs_keys` or `rhs_keys` mismatch or if the
        column names of `lhs_out_columns` and `rhs_out_columns` are not unique.
    """
    if lhs_out_columns is None:
        lhs_out_columns = lhs.get_column_names()
    if rhs_out_columns is None:
        if join_type == JoinType.ANTI or join_type == JoinType.SEMI:
            rhs_out_columns = []  # cannot select any columns with these joins
        else:
            rhs_out_columns = rhs.get_column_names()

    cdef vector[string] lhs_key_vector
    cdef vector[string] rhs_key_vector
    for k in lhs_keys:
        lhs_key_vector.push_back(k.encode('UTF-8'))
    for k in rhs_keys:
        rhs_key_vector.push_back(k.encode('UTF-8'))

    cdef vector[string] lhs_out_columns_vector
    cdef vector[string] rhs_out_columns_vector
    for k in lhs_out_columns:
        lhs_out_columns_vector.push_back(k.encode('UTF-8'))
    for k in rhs_out_columns:
        rhs_out_columns_vector.push_back(k.encode('UTF-8'))

    return LogicalTable.from_handle(
        cpp_join(
            lhs._handle,
            rhs._handle,
            lhs_key_vector,
            rhs_key_vector,
            join_type,
            lhs_out_columns_vector,
            rhs_out_columns_vector,
            nulls_equal,
            broadcast
        )
    )
