# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibcudf.aggregation cimport aggregation as cudf_agg

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

from typing import Iterable, Tuple

from pylibcudf.libcudf.aggregation import Kind as AggregationKind

from legate_dataframe.utils import _track_provenance


# Since Cython doesn't have a `std::tuple` type, we declare a
# specialized one here for the `column_aggregations` argument.
cdef extern from * nogil:
    """
    #include <legate_dataframe/groupby_aggregation.hpp>

    using AggTuple = std::tuple<std::string, cudf::aggregation::Kind, std::string>;
    """

    cdef cppclass AggTuple:
        AggTuple(string, cudf_agg.Kind, string) except +


cdef extern from "<legate_dataframe/groupby_aggregation.hpp>" nogil:
    cpp_LogicalTable cpp_groupby_aggregation \
        "legate::dataframe::groupby_aggregation"(
            const cpp_LogicalTable& table,
            const vector[string]& keys,
            const vector[AggTuple]& column_aggregations
        ) except +


@_track_provenance
def groupby_aggregation(
  LogicalTable table,
  keys: Iterable[str],
  column_aggregations: Iterable[Tuple[str, AggregationKind, str]]
) -> LogicalTable:
    """Perform a groupby and aggregation in a single operation.

    Warning
    -------
    non-default cudf::aggregation arguments are ignored. The default constructor
    is used always. This also means that we only support aggregations that have
    a default constructor!

    Parameters
    ----------
    table
        The table to group and aggregate.
    keys
        The names of the columns whose rows act as the groupby keys.
    column_aggregations
        A list of column aggregations to perform. Each column aggregation produces a
        column in the output table by performing an `AggregationKind` on a column in
        `table`. It consist of a tuple:
        ``(<input-column-name>, <aggregation-kind>, <output-column-name>)``.
        E.g. ``("x", SUM, "sum-of-x")}`` will produce a column named "sum-of-x" in
        the output table, which, for each groupby key, has a row that contains the
        sum of the values in the column "x". Multiple column aggregations can share
        the same input column but all output columns must be unique and not conflict
        with the name of the key columns.

    Returns
    -------
        A new logical table that contains the key columns and the aggregated columns
        using the output column names and order specified in `column_aggregations`.
    """
    cdef vector[string] _keys
    for k in keys:
        _keys.push_back(k.encode('UTF-8'))

    cdef vector[AggTuple] aggs
    for in_col_name, kind, out_col_name in column_aggregations:
        aggs.push_back(
            AggTuple(
                in_col_name.encode('UTF-8'),
                kind.value,
                out_col_name.encode('UTF-8')
            )
        )
    return LogicalTable.from_handle(
        cpp_groupby_aggregation(table._handle, _keys, aggs)
    )
