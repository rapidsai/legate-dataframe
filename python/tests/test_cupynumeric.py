# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest

from legate_dataframe import LogicalColumn, LogicalTable

cn = pytest.importorskip("cupynumeric")


def test_column_round_trip():
    # Testing round-tripping via the legate data interface
    a = cn.arange(100)
    col = LogicalColumn(a)
    b = cn.asarray(col)
    assert cn.array_equal(a, b)


def test_column_interop():
    # Testing round-tripping via the legate data interface
    a = cn.arange(100)
    col = LogicalColumn(a)
    b = cn.add(a, col)
    assert cn.array_equal(b, a + a)


def test_column_to_array():
    # Unlike (currently) the legate data interface our `to_array()` method
    # allows for masks if they are all valid:
    col = LogicalColumn.from_cudf(cudf.Series([1, 2]).mask([False, False])._column)
    arr = col.to_array()
    assert cn.array_equal(arr, [1, 2])
    assert not arr.flags.writeable

    arr = col.to_array(writeable=True)
    assert cn.array_equal(arr, [1, 2])
    assert arr.flags.writeable

    col = LogicalColumn.from_cudf(cudf.Series([None, 2])._column)
    with pytest.raises(ValueError, match=".*that contains NULLs"):
        col.to_array()


def test_column_to_array_bad_dtype():
    col = LogicalColumn.from_cudf(cudf.Series([1, 2], dtype="timedelta64[ns]")._column)
    # Can get the raw array (just to see that it works):
    col.get_logical_array()

    with pytest.raises(TypeError, match=".*not a basic legate type"):
        col.to_array()

    # Also test strings (this fails in cupynumeric only):
    col = LogicalColumn.from_cudf(cudf.Series(["1", "2"])._column)
    with pytest.raises(
        TypeError, match="cupynumeric doesn't support arrays with children"
    ):
        col.to_array()


def test_table_to_array():
    cudf_df = cudf.DataFrame({"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": [4, 5, 6]})
    tbl = LogicalTable.from_cudf(cudf_df)

    arr = tbl.to_array()
    assert arr.dtype == "float64"
    assert cn.array_equal(arr.T, [[1, 2, 3], [2.0, 3.0, 4.0], [4, 5, 6]])

    # Different order and dtype (only central column is float)
    tbl = LogicalTable([tbl["c"], tbl["a"]], ["a", "b"])
    arr = tbl.to_array()
    assert arr.dtype == "int64"
    assert cn.array_equal(arr, [[4, 1], [5, 2], [6, 3]])

    # Check that out= seems to work:
    out = cn.zeros_like(arr)
    arr = tbl.to_array(out=out)
    assert arr is out
    assert cn.array_equal(out, [[4, 1], [5, 2], [6, 3]])
