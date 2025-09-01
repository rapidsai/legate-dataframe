# Copyright (c) 2023-2025, NVIDIA CORPORATION
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
from legate_dataframe.lib.join import BroadcastInput, JoinType, join, null_equality
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import (
    assert_arrow_table_equal,
    assert_matches_polars,
    get_test_scoping,
)


def make_param():
    """Parameters for 'lhs,rhs,left_on,right_on'"""

    for a_num_rows in range(60, 100, 20):
        for b_num_rows in range(60, 100, 20):
            a = np.arange(a_num_rows, dtype="int64")
            b = np.arange(b_num_rows, dtype="int64")
            np.random.shuffle(a)
            np.random.shuffle(b)
            payload_a = pa.array(
                np.arange(a_num_rows),
                mask=np.array([i % 5 == 3 for i in range(a_num_rows)]),
            )
            yield (
                pa.table({"a": a, "payload_a": payload_a}),
                pa.table({"b": b, "payload_b": np.arange(b.size) * -1}),
                ["a"],
                ["b"],
            )
            yield (
                pa.table({"a": a, "payload_a": [str(i) for i in range(a.size)]}),
                pa.table({"b": b, "payload_b": [str(i * -1) for i in range(b.size)]}),
                ["a"],
                ["b"],
            )
    yield (
        pa.table({"a": [1, 2, 3, 4, 5], "payload_a": np.arange(5)}),
        pa.table({"b": [1, 1, 2, 2, 5, 6], "payload_b": np.arange(6) * -1}),
        ["a"],
        ["b"],
    )


@pytest.mark.parametrize(
    "how",
    (
        JoinType.INNER,
        JoinType.LEFT,
        JoinType.FULL,
    ),
)
@pytest.mark.parametrize(
    "broadcast",
    (BroadcastInput.AUTO, BroadcastInput.LEFT, BroadcastInput.RIGHT),
)
@pytest.mark.parametrize("arrow_lhs,arrow_rhs,left_on,right_on", make_param())
@pytest.mark.parametrize("nulls_equal", [True, False])
@pytest.mark.parametrize("scope", get_test_scoping())
def test_join(
    how: JoinType,
    arrow_lhs,
    arrow_rhs,
    left_on,
    right_on,
    broadcast,
    nulls_equal,
    scope,
):
    pl = pytest.importorskip("polars")
    lg_lhs = LogicalTable.from_arrow(arrow_lhs)
    lg_rhs = LogicalTable.from_arrow(arrow_rhs)

    if (how == JoinType.FULL and broadcast != BroadcastInput.AUTO) or (
        how == JoinType.LEFT and broadcast == BroadcastInput.LEFT
    ):
        # In these cases we don't support broadcasting (at least for now)
        with pytest.raises(RuntimeError):
            res = join(
                lg_lhs,
                lg_rhs,
                lhs_keys=left_on,
                rhs_keys=right_on,
                join_type=how,
                broadcast=broadcast,
                compare_nulls=(
                    null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL
                ),
            )
        return

    polars_join = {
        JoinType.INNER: "inner",
        JoinType.LEFT: "left",
        JoinType.FULL: "full",
    }

    expect = (
        pl.from_arrow(arrow_lhs)
        .join(
            pl.from_arrow(arrow_rhs),
            left_on=left_on,
            right_on=right_on,
            how=polars_join[how],
            coalesce=False,
            nulls_equal=nulls_equal,
        )
        .to_arrow()
    )

    with scope:
        res = join(
            lg_lhs,
            lg_rhs,
            lhs_keys=left_on,
            rhs_keys=right_on,
            join_type=how,
            broadcast=broadcast,
            compare_nulls=null_equality.EQUAL if nulls_equal else null_equality.UNEQUAL,
        ).to_arrow()

    sort_order = [(col, "ascending") for col in res.column_names]
    assert_arrow_table_equal(res.sort_by(sort_order), expect.sort_by(sort_order))


def test_column_names_and_strings():
    # Also use string as keys
    lhs = LogicalTable.from_arrow(
        pa.table({"key": ["1", "2", "3"], "data0": [1, 2, 3]})
    )
    rhs = LogicalTable.from_arrow(
        pa.table({"key": ["3", "2", "1"], "data1": ["1", "2", "3"]})
    )

    res = join(
        lhs,
        rhs,
        lhs_keys=["key"],
        rhs_keys=["key"],
        join_type=JoinType.INNER,
        lhs_out_columns=["data0", "key"],
        rhs_out_columns=["data1"],
    )
    expected = pa.table({"data0": [1, 2, 3], "key": [1, 2, 3], "data1": [3, 2, 1]})
    assert_arrow_table_equal(
        res.to_arrow().sort_by([("data0", "ascending"), ("key", "ascending")]),
        expected.sort_by([("data0", "ascending"), ("key", "ascending")]),
    )


@pytest.mark.parametrize("threshold", [0, 2])
def test_empty_chunks(threshold):
    pl = pytest.importorskip("polars")
    # Check that the join code deals gracefully if most/all ranks have no
    # data at all.  `apply_boolean_mask` creates such dataframes.
    values = np.arange(-100, 100)
    # Create a mask that has very few true values in the middle:
    lhs_arrow = pa.table({"a": values, "mask": abs(values) <= threshold})
    lhs_lg_df = LogicalTable.from_arrow(lhs_arrow)

    # Filter using boolean mask
    lhs_lg_df = apply_boolean_mask(lhs_lg_df, lhs_lg_df["mask"])

    # Values exist, but not at the same place:
    rhs_arrow = pa.table({"b": np.arange(0, 200)})
    rhs_lg_df = LogicalTable.from_arrow(rhs_arrow)

    lg_result = join(
        lhs_lg_df,
        rhs_lg_df,
        lhs_keys=["a"],
        rhs_keys=["b"],
        join_type=JoinType.INNER,
    )

    lhs_filtered = pl.from_arrow(lhs_arrow).filter(pl.col("mask"))
    rhs_pl = pl.from_arrow(rhs_arrow)
    expected_result = lhs_filtered.join(
        rhs_pl, left_on=["a"], right_on=["b"], how="inner", coalesce=False
    ).to_arrow()
    assert_arrow_table_equal(
        lg_result.to_arrow().sort_by([("a", "ascending"), ("b", "ascending")]),
        expected_result.sort_by([("a", "ascending"), ("b", "ascending")]),
    )


@pytest.mark.parametrize(
    "how",
    (
        "inner",
        "left",
        "full",
    ),
)
@pytest.mark.parametrize("arrow_lhs,arrow_rhs,left_on,right_on", make_param())
def test_join_basic_polars(how, arrow_lhs, arrow_rhs, left_on, right_on):
    pl = pytest.importorskip("polars")

    lhs = pl.DataFrame(arrow_lhs).lazy()
    rhs = pl.DataFrame(arrow_rhs).lazy()

    q = lhs.join(rhs, left_on=left_on, right_on=right_on, how=how)

    try:
        assert_matches_polars(q.sort(q.columns))
    except ValueError as e:
        # are failing on string columns which are also tested
        if str(e) == "unsupported Arrow datatype":
            pytest.xfail("unsupported Arrow datatype")
        raise
