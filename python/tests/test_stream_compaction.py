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

import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.stream_compaction import apply_boolean_mask, distinct
from legate_dataframe.testing import (
    assert_arrow_table_equal,
    assert_matches_polars,
    get_pyarrow_column_set,
    std_dataframe_set,
)


@pytest.mark.parametrize("df", std_dataframe_set())
def test_apply_boolean_mask_basic(df: pa.Table):
    lg_df = LogicalTable.from_arrow(df)
    np.random.seed(0)
    mask = pa.array(np.random.randint(0, 2, size=len(df)).astype(bool))
    lg_mask = LogicalColumn.from_arrow(mask)

    res = apply_boolean_mask(lg_df, lg_mask)
    expect = df.filter(mask)

    assert_arrow_table_equal(res.to_arrow(), expect)


@pytest.mark.parametrize("df", std_dataframe_set())
def test_apply_boolean_mask_nulls(df: pa.Table):
    # Similar to `test_apply_boolean_mask`, but cover a nullable column
    lg_df = LogicalTable.from_arrow(df)

    np.random.seed(0)
    mask_values = np.random.randint(0, 2, size=len(df)).astype(bool)
    mask_mask = np.random.randint(0, 2, size=len(df)).astype(bool)
    mask = pa.array(mask_values, mask=mask_mask)
    lg_mask = LogicalColumn.from_arrow(mask)

    res = apply_boolean_mask(lg_df, lg_mask)
    expect = df.filter(mask)

    assert_arrow_table_equal(res.to_arrow(), expect)


@pytest.mark.parametrize(
    "bad_mask",
    [
        pa.array([1, 0, 1, 0]),  # not boolean
        # wrong length, but as of writing not caught before at/task launch:
        pytest.param(
            pa.array([True, False, False, True, False]), marks=pytest.mark.skip
        ),
    ],
)
def test_apply_boolean_mask_errors(bad_mask):
    df = pa.table({"a": [1, 2, 3, 4]})

    lg_df = LogicalTable.from_arrow(df)
    bad_mask = LogicalColumn.from_arrow(bad_mask)

    with pytest.raises(ValueError):
        apply_boolean_mask(lg_df, bad_mask)


@pytest.mark.parametrize(
    "keys, high_cardinality",
    [
        (["b", "a"], True),
        (["a", "b"], False),
        (["b", "c"], False),
        (None, False),
    ],
)
def test_distinct_basic(keys, high_cardinality):
    df = pa.table(
        {"a": [1, 1, 2, 3, 4], "b": [2, 2, 4, 4, None], "c": [10, 10, 20, 30, 40]}
    )
    lg_df = LogicalTable.from_arrow(df)
    res = distinct(lg_df, keys, high_cardinality=high_cardinality)
    expect = pa.table({"a": [1, 2, 3, 4], "b": [2, 4, 4, None], "c": [10, 20, 30, 40]})

    # Result has no defined order, so need to sort before comparing
    sort_by = [("a", "ascending"), ("b", "ascending"), ("c", "ascending")]
    assert_arrow_table_equal(res.to_arrow().sort_by(sort_by), expect.sort_by(sort_by))


@pytest.mark.parametrize(
    "arrow_column", get_pyarrow_column_set(["int32", "float32", "int64"])
)
def test_column_filter_polars(arrow_column):
    pl = pytest.importorskip("polars")

    q = pl.DataFrame({"a": arrow_column}).lazy()
    q = q.filter(pl.col("a") > 0.5)

    assert_matches_polars(q)


@pytest.mark.parametrize("dtype", ["float64", "int32", "int64"])
@pytest.mark.parametrize("keys", [["a", "b"], ["a"], ["b"], None])
def test_polars_distinct(dtype, keys):
    pl = pytest.importorskip("polars")

    mask = np.random.randint(2, size=10_000, dtype=bool)
    a = np.random.randint(100, size=10_000).astype(dtype)
    if dtype == "float64":
        a[[200, 500, 600]] = np.nan  # see how NaNs are handled.

    b = -a  # Don't deal with duplicates, since we can get any value back
    b[mask] = 1

    q = pl.DataFrame(
        {
            "a": pa.array(a, mask=mask),
            "b": b,
        }
    ).lazy()

    # NOTE(seberg): Descending sort because NaN sorting behaves differently for arrow
    # (That may need working around in sort; as of 2025-10.)
    assert_matches_polars(q.unique(keys).sort(["a", "b"], descending=True))
