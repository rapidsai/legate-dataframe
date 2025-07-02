# Copyright (c) 2025, NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import cupy
import pytest

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import (
    assert_frame_equal,
    assert_matches_polars,
    get_pyarrow_column_set,
    std_dataframe_set,
)


@pytest.mark.parametrize("cudf_df", std_dataframe_set())
def test_apply_boolean_mask_basic(cudf_df: cudf.DataFrame):
    lg_df = LogicalTable.from_cudf(cudf_df)

    cupy.random.seed(0)
    cudf_mask = cudf.Series(cupy.random.randint(0, 2, size=len(cudf_df)), dtype=bool)
    lg_mask = LogicalColumn.from_cudf(cudf_mask._column)

    res = apply_boolean_mask(lg_df, lg_mask)
    expect = cudf_df[cudf_mask]

    assert_frame_equal(res, expect)


@pytest.mark.parametrize("cudf_df", std_dataframe_set())
def test_apply_boolean_mask_nulls(cudf_df: cudf.DataFrame):
    # Similar to `test_apply_boolean_mask`, but cover a nullable column
    lg_df = LogicalTable.from_cudf(cudf_df)

    cupy.random.seed(0)
    mask_values = cupy.random.randint(0, 2, size=len(cudf_df)).astype(bool)
    mask_mask = cupy.random.randint(0, 2, size=len(cudf_df)).astype(bool)
    cudf_mask = cudf.Series(mask_values)
    cudf_mask = cudf_mask.mask(mask_mask)
    lg_mask = LogicalColumn.from_cudf(cudf_mask._column)

    res = apply_boolean_mask(lg_df, lg_mask)
    expect = cudf_df[cudf_mask]

    assert_frame_equal(res, expect)


@pytest.mark.parametrize(
    "bad_mask",
    [
        cudf.Series([1, 2, 3, 4]),  # not boolean
        # wrong length, but as of writing not caught before at/task launch:
        pytest.param(
            cudf.Series([True, False, False, True, False]), marks=pytest.mark.skip
        ),
    ],
)
def test_apply_boolean_mask_errors(bad_mask):
    df = cudf.DataFrame({"a": [1, 2, 3, 4]})

    lg_df = LogicalTable.from_cudf(df)
    bad_mask = LogicalColumn.from_cudf(bad_mask._column)

    with pytest.raises(ValueError):
        apply_boolean_mask(lg_df, bad_mask)


@pytest.mark.parametrize(
    "arrow_column", get_pyarrow_column_set(["int32", "float32", "int64"])
)
def test_column_replace_null_polars(arrow_column):
    pl = pytest.importorskip("polars")

    q = pl.DataFrame({"a": arrow_column}).lazy()
    q = q.filter(pl.col("a") > 0.5)

    assert_matches_polars(q)
