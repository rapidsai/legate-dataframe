# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from legate.core import StoreTarget, get_legate_runtime
from testing import try_import_cudf

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.unaryop import unary_operation
from legate_dataframe.testing import get_pyarrow_column_set, guess_available_mem


def test_column_name_by_index():
    df = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    tbl = LogicalTable.from_arrow(df)

    assert df["a"].combine_chunks().equals(tbl.get_column(0).to_arrow())
    assert df["b"].combine_chunks().equals(tbl.get_column(1).to_arrow())
    assert tbl.get_column_names() == ["a", "b"]
    assert df["a"].combine_chunks().equals(tbl[0].to_arrow())
    assert df["b"].combine_chunks().equals(tbl[1].to_arrow())

    with pytest.raises(IndexError):
        tbl.get_column(2)

    with pytest.raises(OverflowError):
        tbl.get_column(-1)

    with pytest.raises(TypeError):
        tbl.get_column(object())


def test_column_name_by_string():
    df = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    tbl = LogicalTable.from_arrow(df)
    assert df["a"].combine_chunks().equals(tbl.get_column("a").to_arrow())
    assert df["b"].combine_chunks().equals(tbl.get_column("b").to_arrow())
    assert tbl.get_column_names() == ["a", "b"]
    assert df["a"].combine_chunks().equals(tbl["a"].to_arrow())
    assert df["b"].combine_chunks().equals(tbl["b"].to_arrow())
    with pytest.raises(IndexError):
        tbl.get_column("c")


@pytest.mark.parametrize(
    "array",
    get_pyarrow_column_set(["int32", "float32", "timestamp[s]", "int64"]),
)
def test_column_dtype(array):
    col = LogicalColumn.from_arrow(array)
    assert col.dtype() == array.type


@pytest.mark.skip(reason="Test is fairly slow and requires a lot of GPU memory.")
@pytest.mark.parametrize("size", [2**31, 2**31 + 16])
def test_huge_string_roundtrip(size):
    cudf = try_import_cudf()
    # Sanity check that round-tripping huge string columns also works:
    col = cudf.Series([12345678, 23456789], dtype="int32").astype(str)
    # The above has a string size of more than 8 * 2 bytes. Repeat it to be
    # too large (as of 24.06 requires env variable to be set)
    col_cudf = col.repeat(size // (8 * 2))

    # For clarity, the length of each element is 8 bytes so the total size
    # requires long string support:
    assert len(col_cudf) * 8 == size

    # Check that round-tripping works.
    col_lg = LogicalColumn.from_cudf(col_cudf._column)

    # Compare via arrow (CPU) to reduce memory pressure
    col_cudf_arrow = col_cudf.to_arrow()
    del col_cudf
    col_lg_arrow = col_lg.to_cudf().to_arrow()
    del col_lg

    assert col_cudf_arrow == col_lg_arrow


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
def test_column_slice(slice_):
    arr = pa.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mask=[False, True] * 5,
    )
    expected = arr[slice_]

    lg_col = LogicalColumn.from_arrow(arr)

    assert lg_col[slice_].to_arrow() == expected
    assert lg_col.slice(slice_).to_arrow() == expected


@pytest.mark.skip(reason="This causes CI hangs. Investigate rewriting this test.")
def test_offload_to():
    cudf = try_import_cudf()
    # Note that, if `LEGATE_CONFIG` is set but not used, this may currently fail.
    available_mem_gpu, available_mem_cpu = guess_available_mem()
    if not available_mem_gpu or not available_mem_cpu:
        pytest.skip(reason="Could not guess available GPU or SYSMEM.")
    if available_mem_cpu < available_mem_gpu * 2.5:
        pytest.skip(reason="Need a more SYSMEM than GPU mem for test.")

    length = available_mem_gpu // 10 // 8 * 1024**2
    col = cudf.Series([1], dtype="int64")
    col = col.repeat(length)
    col_lg = LogicalColumn.from_cudf(col._column)

    results = []
    for i in range(15):
        # Taking the negative 20 times can't possibly fit into GPU memory
        res = unary_operation(col_lg, "abs")
        # but should work if we offload all results
        res.offload_to(StoreTarget.SYSMEM)
        results.append(res)

        # Make sure we clean up before we continue (or finalize the program)
        # (As of writing, doing it every time prevents a hang.)
        get_legate_runtime().issue_execution_fence(block=True)

    # Not sure if helpful, but delete and wait.
    del col_lg, results
    get_legate_runtime().issue_execution_fence(block=True)
