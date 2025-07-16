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
from legate.core import get_legate_runtime

from legate_dataframe import LogicalTable
from legate_dataframe.lib.sort import sort
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import assert_arrow_table_equal


@pytest.mark.parametrize(
    "values",
    [
        np.arange(0, 1000),
        np.arange(0, -1000, -1),
        np.ones(1000),
        np.ones(1),
        np.random.randint(0, 1000, size=1000),
    ],
)
def test_basic(values):
    df = pa.table({"a": values})

    lg_df = LogicalTable.from_arrow(df)
    lg_sorted = sort(lg_df, ["a"])

    df_sorted = df.sort_by("a")

    assert_arrow_table_equal(lg_sorted.to_arrow(), df_sorted)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(0, 1000),
        np.arange(0, -1000, -1),
        np.ones(1000),
        np.ones(3),
        np.random.randint(0, 1000, size=1000),
    ],
)
def test_basic_with_extra_column(values):
    df = pa.table({"a": values, "b": np.arange(len(values))})

    lg_df = LogicalTable.from_arrow(df)
    lg_sorted = sort(lg_df, ["a"])

    df_sorted = df.sort_by("a")

    assert_arrow_table_equal(lg_sorted.to_arrow(), df_sorted)


@pytest.mark.parametrize("threshold", [0, 2])
def test_empty_chunks(threshold):
    # The sorting code needs to be careful when some ranks have zero rows.
    # In that case we the rank has no split points to share and the total number
    # of split points may be fewer than the number of ranks.
    values = np.arange(-100, 100)
    # Create a mask that has very few true values in the middle:
    df = pa.table({"a": values, "mask": abs(values) <= threshold})
    lg_df = LogicalTable.from_arrow(df)

    lg_result = sort(apply_boolean_mask(lg_df, lg_df["mask"]), ["a"])

    # Filter and sort the arrow table
    df_filtered = df.filter(df.column("mask"))
    df_result = df_filtered.sort_by("a")

    assert_arrow_table_equal(lg_result.to_arrow(), df_result)


@pytest.mark.parametrize("reversed", [True, False])
def test_shifted_equal_window(reversed):
    # The tricky part about sorting are the exact splits for exchanging.
    # assume we have at least two gpus/workers.  Shift a window of 50
    # (i.e. half of each worker), through, to see if it gets split incorrectly.
    for i in range(150):
        before = np.arange(i)
        constant = np.full(50, i)
        after = np.arange(50 + i, 200)
        values = np.concatenate([before, constant, after])
        if reversed:
            values = values[::-1].copy()

        # Need a second column to check the splits:
        df = pa.table({"a": values, "b": np.arange(200)})

        lg_df = LogicalTable.from_arrow(df)
        lg_sorted = sort(lg_df, ["a"], stable=True)
        df_sorted = df.sort_by("a")

        assert_arrow_table_equal(lg_sorted.to_arrow(), df_sorted)

        # Block for stability with lower memory (not sure if it should be here)
        get_legate_runtime().issue_execution_fence(block=True)


@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize(
    "by,ascending,nulls_last",
    [
        (["a"], [True], True),  # completely standard sort
        (["a"], [False], False),
        (["a", "b", "c"], [True, False, True], True),
        (["c", "a", "b"], [True, False, True], False),
        (["c", "b", "a"], [True, False, True], True),
    ],
)
def test_orders(by, ascending, nulls_last, stable):
    # Note that Arrow sort_indices doesn't allow passing null_placement as a list.
    # So we'll test with simple cases for now that match the current sort API
    np.random.seed(1)

    if not stable:
        # If the sort is not stable, include index to have stable results...
        by.append("idx")
        ascending.append(True)

    # Generate a dataset with many repeats so all columns should matter
    # repeats = 100
    repeats = 1
    values_a = np.arange(10).repeat(repeats)
    values_b = np.arange(10.0).repeat(repeats)
    values_c = ["a", "b", "hello", "d", "e", "f", "e", "🙂", "e", "g"] * repeats

    np.random.shuffle(values_a)
    np.random.shuffle(values_b)

    # Create arrays with nulls using numpy
    null_mask_a = np.random.choice([True, False], size=values_a.size, p=[0.1, 0.9])
    values_a_with_nulls = values_a.astype(float)
    values_a_with_nulls[null_mask_a] = np.nan

    null_mask_b = np.random.choice([True, False], size=values_a.size, p=[0.1, 0.9])
    values_b_with_nulls = values_b.copy()
    values_b_with_nulls[null_mask_b] = np.nan

    null_mask_c = np.random.choice([True, False], size=values_a.size, p=[0.1, 0.9])
    values_c_with_nulls = np.array(values_c, dtype=object)
    values_c_with_nulls[null_mask_c] = None

    arrow_table = pa.table(
        {
            "a": pa.array(values_a_with_nulls),
            "b": pa.array(values_b_with_nulls),
            "c": pa.array(values_c_with_nulls),
            "idx": np.arange(values_a.size),
        }
    )
    lg_df = LogicalTable.from_arrow(arrow_table)

    # Arrow sort_by expects individual sort keys with their own order
    sort_keys = []
    for i, key in enumerate(by):
        order = "ascending" if ascending[i] else "descending"
        sort_keys.append((key, order))

    expected = arrow_table.sort_by(
        sort_keys, null_placement="at_end" if nulls_last else "at_start"
    )

    # Use the current sort API which takes sort_ascending and nulls_at_end
    lg_sorted = sort(
        lg_df,
        keys=by,
        sort_ascending=ascending,
        nulls_at_end=nulls_last,
        stable=stable,
    )

    assert_arrow_table_equal(lg_sorted.to_arrow(), expected)


def test_na_position_explicit():
    # Create Arrow table with nulls
    arrow_table = pa.table(
        {
            "a": pa.array([0, 1, None, None], type=pa.int64()),
            "b": pa.array([1, None, 0, None], type=pa.int64()),
        }
    )

    lg_df = LogicalTable.from_arrow(arrow_table)
    lg_sorted = sort(lg_df, ["a", "b"], nulls_at_end=False)

    expected = pa.table(
        {
            "a": pa.array([None, None, 0, 1], type=pa.int64()),
            "b": pa.array([None, 0, 1, None], type=pa.int64()),
        }
    )

    assert_arrow_table_equal(lg_sorted.to_arrow(), expected)


@pytest.mark.parametrize(
    "keys,sort_ascending,nulls_at_end",
    [
        ([], None, None),  # Empty keys should fail
        (["bad_col"], None, None),  # Non-existent column should fail
        (["a"], [True, False], None),  # Mismatched keys and sort_ascending length
        (["a", "b"], [True], None),  # Mismatched keys and sort_ascending length
    ],
)
def test_errors_incorrect_args(keys, sort_ascending, nulls_at_end):
    arrow_table = pa.table({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    lg_df = LogicalTable.from_arrow(arrow_table)

    with pytest.raises((ValueError, TypeError, KeyError)):
        sort(lg_df, keys=keys, sort_ascending=sort_ascending, nulls_at_end=nulls_at_end)
