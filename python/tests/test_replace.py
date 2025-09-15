# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.replace import replace_nulls
from legate_dataframe.testing import assert_matches_polars, get_pyarrow_column_set


@pytest.mark.parametrize(
    "column", get_pyarrow_column_set(["int32", "float32", "int64"])
)
def test_column_replace_null(column):
    col = LogicalColumn.from_arrow(column)

    expected = pa.compute.fill_null(column, 1)
    res = replace_nulls(col, pa.scalar(1, type=column.type))

    assert expected.equals(res.to_arrow())


@pytest.mark.parametrize(
    "column", get_pyarrow_column_set(["int32", "float32", "int64"])
)
def test_column_replace_null_with_null(column):
    # Replacing with NULL is odd, but at least tests passing NULLs to tasks.
    col = LogicalColumn.from_arrow(column)
    value = pa.scalar(None, type=column.type)
    res = replace_nulls(col, value)
    # The result should be the same as the input
    assert column.equals(res.to_arrow())


@pytest.mark.parametrize(
    "arrow_column", get_pyarrow_column_set(["int32", "float32", "int64"])
)
def test_column_replace_null_polars(arrow_column):
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"a": arrow_column}).lazy()
    q = df.with_columns(filled=pl.col("a").fill_null(1))
    assert_matches_polars(q)
