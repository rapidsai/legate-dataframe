# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.testing import assert_frame_equal, get_column_set


def test_column_name_by_index():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    tbl = LogicalTable.from_cudf(df)

    assert_frame_equal(df["a"]._column, tbl.get_column(0).to_cudf())
    assert_frame_equal(df["b"]._column, tbl.get_column(1).to_cudf())
    assert tbl.get_column_names() == ["a", "b"]
    assert_frame_equal(df["a"]._column, tbl[0].to_cudf())
    assert_frame_equal(df["b"]._column, tbl[1].to_cudf())

    with pytest.raises(IndexError):
        tbl.get_column(2)

    with pytest.raises(OverflowError):
        tbl.get_column(-1)

    with pytest.raises(TypeError):
        tbl.get_column(object())


def test_column_name_by_string():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    tbl = LogicalTable.from_cudf(df)

    assert_frame_equal(df["a"]._column, tbl.get_column("a").to_cudf())
    assert_frame_equal(df["b"]._column, tbl.get_column("b").to_cudf())
    assert tbl.get_column_names() == ["a", "b"]
    assert_frame_equal(df["a"]._column, tbl["a"].to_cudf())
    assert_frame_equal(df["b"]._column, tbl["b"].to_cudf())

    with pytest.raises(IndexError):
        tbl.get_column("c")


@pytest.mark.parametrize(
    "cudf_column",
    get_column_set(["int32", "float32", "M8[s]", "int64"]),
)
def test_column_dtype(cudf_column):
    col = LogicalColumn.from_cudf(cudf_column)
    assert col.cudf_dtype() == cudf_column.dtype


@pytest.mark.skip(reason="Test is fairly slow and requires a lot of GPU memory.")
@pytest.mark.parametrize("size", [2**31, 2**31 + 16])
def test_huge_string_roundtrip(size):
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
