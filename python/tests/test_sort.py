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


"""

@pytest.mark.parametrize("reversed", [True, False])
def test_shifted_equal_window(reversed):
    # The tricky part abort sorting are the exact splits for exchanging.
    # assume we have at least two gpus/workders.  Shift a window of 50
    # (i.e. half of each worker), through, to see if it gets split incorrectly.
    for i in range(150):
        before = np.arange(i)
        constant = np.full(50, i)
        after = np.arange(50 + i, 200)
        values = np.concatenate([before, constant, after])
        if reversed:
            values = values[::-1].copy()

        # Need a second column to check the splits:
        df = pd.DataFrame({"a": values, "b": np.arange(200)})

        lg_df = LogicalTable.from_pandas(df)
        lg_sorted = sort(lg_df, ["a"], stable=True)
        df_sorted = df.sort_values(by=["a"], kind="stable")

        assert_frame_equal(lg_sorted, df_sorted)

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
    # Note that pandas sort_values doesn't allow passing na_position as a list.
    # So we'll test with simple cases for now that match the current sort API
    np.random.seed(1)

    if not stable:
        # If the sort is not stable, include index to have stable results...
        by.append("idx")
        ascending.append(True)

    # Generate a dataset with many repeats so all columns should matter
    values_a = np.arange(10).repeat(100)
    values_b = np.arange(10.0).repeat(100)
    values_c = ["a", "b", "hello", "d", "e", "f", "e", "🙂", "e", "g"] * 100

    np.random.shuffle(values_a)
    np.random.shuffle(values_b)

    # Create series with nulls using pandas
    series_a = pd.Series(values_a)
    series_a[np.random.choice([True, False], size=1000, p=[0.1, 0.9])] = np.nan

    series_b = pd.Series(values_b)
    series_b[np.random.choice([True, False], size=1000, p=[0.1, 0.9])] = np.nan

    series_c = pd.Series(values_c)
    series_c[np.random.choice([True, False], size=1000, p=[0.1, 0.9])] = None

    pandas_df = pd.DataFrame(
        {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "idx": np.arange(1000),
        }
    )
    lg_df = LogicalTable.from_pandas(pandas_df)

    kind = "stable" if stable else "quicksort"
    na_position = "last" if nulls_last else "first"
    expected = pandas_df.sort_values(
        by=by, ascending=ascending, na_position=na_position, kind=kind
    )

    # Use the current sort API which takes sort_ascending and nulls_at_end
    lg_sorted = sort(
        lg_df,
        keys=by,
        sort_ascending=ascending,
        nulls_at_end=nulls_last,
        stable=stable,
    )

    assert_frame_equal(lg_sorted, expected)


def test_na_position_explicit():
    pandas_df = pd.DataFrame({"a": [0, 1, None, None], "b": [1, None, 0, None]})

    lg_df = LogicalTable.from_pandas(pandas_df)
    # Test with nulls_at_end=False (nulls at beginning)
    lg_sorted = sort(lg_df, ["a", "b"], nulls_at_end=False)

    expected = pd.DataFrame({"a": [None, None, 0, 1], "b": [0, None, 1, None]})

    assert_frame_equal(lg_sorted, expected)


@pytest.mark.parametrize(
    "keys,column_order,null_precedence",
    [
        ([], None, None),
        (["bad_col", None, None]),
        (["a"], [Order.ASCENDING] * 2, None),
        (["a"], None, [NullOrder.BEFORE] * 2),
        # These should fail (wrong enum passed), but cython doesn't check:
        # (["a", "b"], [Order.ASCENDING] * 2, [Order.ASCENDING] * 2),
        # (["a", "b"], [NullOrder.BEFORE] * 2, [NullOrder.BEFORE] * 2),
    ],
)
def test_errors_incorrect_args(keys, column_order, null_precedence):
    df = cudf.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    lg_df = LogicalTable.from_cudf(df)

    with pytest.raises((ValueError, TypeError)):
        sort(
            lg_df, keys=keys, column_order=column_order, null_precedence=null_precedence
        )
"""
