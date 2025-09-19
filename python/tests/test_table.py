# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from legate.core import StoreTarget, get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import (
    assert_arrow_table_equal,
    assert_matches_polars,
    guess_available_mem,
    try_import_cudf,
)


@pytest.mark.skip(reason="This causes CI hangs. Investigate rewriting this test.")
def test_offload_to():
    cudf = try_import_cudf()
    # Note that, if `LEGATE_CONFIG` is set but not used, this may currently fail.
    available_mem_gpu, available_mem_cpu = guess_available_mem()
    if not available_mem_gpu or not available_mem_cpu:
        pytest.skip(reason="Could not guess available GPU or SYSMEM.")
    if available_mem_cpu < available_mem_gpu * 2.5:
        pytest.skip(reason="Need a more SYSMEM than GPU mem for test.")

    length = available_mem_gpu // 10 * 1024**2
    col = cudf.Series([True], dtype="bool")
    col = col.repeat(length)
    col_lg = LogicalColumn.from_cudf(col._column)
    tbl = LogicalTable([col_lg], "a")

    results = []
    for i in range(15):
        # Taking the negative 15 times can't possibly fit into GPU memory
        res = apply_boolean_mask(tbl, col_lg)
        # but should work if we offload all results
        res.offload_to(StoreTarget.SYSMEM)
        results.append(res)

        # Make sure we clean up before we continue (or finalize the program)
        # (As of writing, doing it every time prevents a hang.)
        get_legate_runtime().issue_execution_fence(block=True)

    # Not sure if helpful, but delete and wait.
    del col_lg, tbl, results
    get_legate_runtime().issue_execution_fence(block=True)


@pytest.mark.parametrize(
    "slice_",
    [
        slice(None),
        slice(0, 3),
        slice(3, 7),
        slice(-8, -2),
        slice(1, 1),
        slice(2, None),
        slice(None, 8),
    ],
)
def test_table_slice(slice_):
    table = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "b": pa.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask=[False, True] * 5),
        }
    )
    lg_table = LogicalTable.from_arrow(table)

    start, stop, _ = slice_.indices(lg_table.num_rows())
    expected = table.slice(start, stop - start)
    res = lg_table.slice(slice_)
    assert_arrow_table_equal(res.to_arrow(), expected)


def test_table_slice_polars():
    pl = pytest.importorskip("polars")

    q = pl.DataFrame(
        {
            "a": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "b": pa.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask=[False, True] * 5),
        }
    ).lazy()
    # Add an "unused" head call to prevent polars pushing down the slice
    # (right now this works, it's plausible to change...)
    q = q.head(9)

    assert_matches_polars(q.slice(1, 4))
    assert_matches_polars(q.slice(-5, 3))
    assert_matches_polars(q.tail(5))
    assert_matches_polars(q.head(7))


@pytest.mark.parametrize(
    "cols", [["a", "b"], ["c", "a"], ["c", "b"], ["c"], [2, 1], [2, 0], [1]]
)
def test_select_and_getitem_table(cols):
    df = pa.table({"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": [4, 5, 6]})
    tbl = LogicalTable.from_arrow(df)

    if isinstance(cols[0], str):
        expected_names = cols
    else:
        expected_names = [["a", "b", "c"][i] for i in cols]

    res = tbl.select(cols)
    assert res.get_column_names() == expected_names
    assert_arrow_table_equal(res.to_arrow(), df.select(expected_names))

    res = tbl[cols]
    assert res.get_column_names() == expected_names
    assert_arrow_table_equal(res.to_arrow(), df.select(expected_names))


def test_select_and_getitem_table_empty():
    df = pa.table({"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": [4, 5, 6]})
    tbl = LogicalTable.from_arrow(df)

    assert tbl[[]].num_columns() == 0


@pytest.mark.parametrize("cols", [(1, 2), ["a", 1], [None]])
def test_select_and_getitem_table_errors(cols):
    # Test type errors (via `[]` indexing, which also rejects non-lists).
    df = pa.table({"a": [1, 2, 3], "b": [2.0, 3.0, 4.0], "c": [4, 5, 6]})
    tbl = LogicalTable.from_arrow(df)

    with pytest.raises(TypeError):
        tbl[cols]


def test_to_array_nullable():
    cn = pytest.importorskip("cupynumeric")

    table = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "b": pa.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask=[False] * 10),
        }
    )
    # NOTE: Ideally, `b` will be nullable (it probably is not), we check the error below though
    lg_table = LogicalTable.from_arrow(table)

    res = lg_table.to_array()
    expected = cn.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
        ]
    )
    assert (res == expected).all()
    assert res.dtype == "int64"

    # Also check that we see when there are masked values and give an error:
    table = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "b": pa.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask=[True, False] * 5),
        }
    )
    lg_table = LogicalTable.from_arrow(table)
    with pytest.raises(ValueError, match=".*contains NULLs to cupynumeric"):
        lg_table.to_array()


def test_polars_lazy_roundtrip_basic():
    # Mainly test that the `.lazy()` method works as expected when collecting
    # with `.legate.collect()`.
    pl = pytest.importorskip("polars")
    from legate_dataframe import ldf_polars  # noqa: F401

    polars_df = (
        pl.DataFrame(
            {
                "a": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "b": pa.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mask=[False, True] * 5),
            }
        )
        .lazy()
        .legate.collect()
    )

    legate_df = polars_df.lazy().legate.collect()
    round_trip = legate_df.lazy().legate.collect()

    assert_arrow_table_equal(round_trip.to_arrow(), legate_df.to_arrow())
