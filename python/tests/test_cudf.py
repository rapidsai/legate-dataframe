# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest
from cudf.testing.testing import assert_column_equal

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.testing import assert_frame_equal, std_dataframe_set


@pytest.mark.parametrize("cudf_col", [cudf.DataFrame({"a": range(10)})["a"]])
def test_column_round_trip(cudf_col):
    # We support cudf and pylibcudf columns (and series) but return columns
    cmp_col = cudf_col._column
    for col in [cudf_col, cudf_col._column, cudf_col._column.to_pylibcudf("read")]:
        col = LogicalColumn.from_cudf(cudf_col)
        cudf_res = col.to_cudf()

        assert not col.is_scalar()
        assert_column_equal(cmp_col, cudf_res)


def test_scalar_column_round_trip():
    cudf_scalar = cudf.Scalar(3)
    # We support both cudf and pylibcudf scalars (but return cudf ones)
    for scalar in [cudf_scalar, cudf_scalar.device_value]:
        col = LogicalColumn.from_cudf(scalar)
        assert col.is_scalar()

        cudf_res = col.to_cudf_scalar()
        assert cudf_res.value == cudf_scalar.value


def test_non_scalar_column_error():
    cudf_col = cudf.Series(range(10))._columns[0]
    col = LogicalColumn.from_cudf(cudf_col)

    with pytest.raises(ValueError, match="only length 1/scalar columns"):
        col.to_cudf_scalar()


@pytest.mark.parametrize("df", std_dataframe_set())
def test_table_round_trip(df):
    tbl = LogicalTable.from_cudf(df)
    res = tbl.to_cudf()
    assert_frame_equal(df, res)
