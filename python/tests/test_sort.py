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
import numpy as np
import pytest

from legate_dataframe import LogicalTable
from legate_dataframe.lib.sort import NullOrder, Order, sort
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import assert_frame_equal


@pytest.mark.parametrize(
    "values",
    [
        cupy.arange(0, 1000),
        cupy.arange(0, -1000, -1),
        cupy.ones(1000),
        cupy.ones(1),
        cupy.random.randint(0, 1000, size=1000),
    ],
)
def test_basic(values):
    df = cudf.DataFrame({"a": values})

    lg_df = LogicalTable.from_cudf(df)
    lg_sorted = sort(lg_df, ["a"])

    df_sorted = df.sort_values(by=["a"])

    assert_frame_equal(lg_sorted, df_sorted)


@pytest.mark.parametrize(
    "values,stable",
    [
        (cupy.arange(0, 1000), False),
        (cupy.arange(0, 1000), True),
        (cupy.arange(0, -1000, -1), False),
        (cupy.arange(0, -1000, -1), True),
        (cupy.ones(1000), True),
        (cupy.ones(3), True),
        (cupy.random.randint(0, 1000, size=1000), True),
    ],
)
def test_basic_with_extra_column(values, stable):
    # Similar as above, but additional column should stay shuffle same.
    df = cudf.DataFrame({"a": values, "b": cupy.arange(len(values))})

    lg_df = LogicalTable.from_cudf(df)
    lg_sorted = sort(lg_df, ["a"], stable=stable)

    if not stable:
        df_sorted = df.sort_values(by=["a"])
    else:
        df_sorted = df.sort_values(by=["a"], kind="stable")

    assert_frame_equal(lg_sorted, df_sorted)


@pytest.mark.parametrize("threshold", [0, 2])
def test_empty_chunks(threshold):
    # The sorting code needs to be careful when some ranks have zero rows.
    # In that case we the rank has no split points to share and the total number
    # of split points may be fewer than the number of ranks.
    values = cupy.arange(-100, 100)
    # Create a mask that has very few true values in the middle:
    df = cudf.DataFrame({"a": values, "mask": abs(values) <= threshold})
    lg_df = LogicalTable.from_cudf(df)

    lg_result = sort(apply_boolean_mask(lg_df, lg_df["mask"]), ["a"])
    df_result = df[df["mask"]].sort_values(by=["a"])

    assert_frame_equal(lg_result, df_result)


@pytest.mark.parametrize("reversed", [True, False])
def test_shifted_equal_window(reversed):
    # The tricky part abort sorting are the exact splits for exchanging.
    # assume we have at least two gpus/workders.  Shift a window of 50
    # (i.e. half of each worker), through, to see if it gets split incorrectly.
    for i in range(150):
        before = cupy.arange(i)
        constant = cupy.full(50, i)
        after = cupy.arange(50 + i, 200)
        values = cupy.concatenate([before, constant, after])
        if reversed:
            values = values[::-1].copy()

        # Need a second column to check the splits:
        df = cudf.DataFrame({"a": values, "b": cupy.arange(200)})

        lg_df = LogicalTable.from_cudf(df)
        lg_sorted = sort(lg_df, ["a"], stable=True)
        df_sorted = df.sort_values(by=["a"], kind="stable")

        assert_frame_equal(lg_sorted, df_sorted)


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
    # Note that cudf/pandas don't allow passing na_position as a list.
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
    series_a = cudf.Series(values_a).mask(
        np.random.choice([True, False], size=1000, p=[0.1, 0.9])
    )
    series_b = cudf.Series(values_b).mask(
        np.random.choice([True, False], size=1000, p=[0.1, 0.9])
    )
    series_c = cudf.Series(values_c).mask(
        np.random.choice([True, False], size=1000, p=[0.1, 0.9])
    )

    cudf_df = cudf.DataFrame(
        {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "idx": cupy.arange(1000),
        }
    )
    lg_df = LogicalTable.from_cudf(cudf_df)

    kind = "stable" if stable else "quicksort"
    na_position = "last" if nulls_last else "first"
    expected = cudf_df.sort_values(
        by=by, ascending=ascending, na_position=na_position, kind=kind
    )

    column_order = [Order.ASCENDING if a else Order.DESCENDING for a in ascending]
    # If nulls are last they are considered "after" for an ascending sort, but
    # if nulls come first they are considered "before"/smaller all values:
    if nulls_last:
        null_precedence = [
            NullOrder.AFTER if a else NullOrder.BEFORE for a in ascending
        ]
    else:
        null_precedence = [
            NullOrder.BEFORE if a else NullOrder.AFTER for a in ascending
        ]

    lg_sorted = sort(
        lg_df,
        keys=by,
        column_order=column_order,
        null_precedence=null_precedence,
        stable=stable,
    )

    assert_frame_equal(lg_sorted, expected)


def test_na_position_explicit():
    cudf_df = cudf.DataFrame({"a": [0, 1, None, None], "b": [1, None, 0, None]})

    lg_df = LogicalTable.from_cudf(cudf_df)
    lg_sorted = sort(
        lg_df, ["a", "b"], null_precedence=[NullOrder.BEFORE, NullOrder.AFTER]
    )

    expected = cudf.DataFrame({"a": [None, None, 0, 1], "b": [0, None, 1, None]})

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
