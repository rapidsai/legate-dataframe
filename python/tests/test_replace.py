# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.replace import replace_nulls
from legate_dataframe.testing import assert_frame_equal, get_column_set


@pytest.mark.parametrize(
    "cudf_column", get_column_set(["int32", "float32", "M8[s]", "int64"])
)
def test_column_replace_null(cudf_column):
    col = LogicalColumn.from_cudf(cudf_column)

    expected = cudf_column.fillna(1)
    res = replace_nulls(col, cudf.Scalar(1, dtype=cudf_column.dtype))

    assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "cudf_column", get_column_set(["int32", "float32", "M8[s]", "int64"])
)
def test_column_replace_null_with_null(cudf_column):
    # Replacing with NULL is odd, but at least tests passing NULLs to tasks.
    col = LogicalColumn.from_cudf(cudf_column)
    value = cudf.Scalar(None, dtype=cudf_column.dtype)

    res = replace_nulls(col, value)
    # The result should be the same as the input
    assert_frame_equal(res, col)
