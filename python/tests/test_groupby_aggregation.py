# Copyright (c) 2024-2025, NVIDIA CORPORATION
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


from typing import Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalTable
from legate_dataframe.lib.groupby_aggregation import groupby_aggregation
from legate_dataframe.testing import assert_arrow_table_equal, assert_matches_polars


def arrow_groupby(
    table: pa.Table,
    keys: List[str],
    column_aggregations: Iterable[Tuple[str, str, str]],
) -> pa.Table:
    """Helper function that performs Arrow groupby using the legate syntax"""

    pyarrow_aggregations = [(a, b) for a, b, _ in column_aggregations]
    for i in range(len(pyarrow_aggregations)):
        if pyarrow_aggregations[i][1] == "count_all":
            # count_all is a special case, it has no input column
            pyarrow_aggregations[i] = ([], "count_all")

    result = pa.TableGroupBy(table, keys).aggregate(pyarrow_aggregations)

    # rename the aggregations according to the given names
    names = keys.copy()

    for _, _, out_name in column_aggregations:
        names.append(out_name)
    return result.rename_columns(names)


@pytest.mark.parametrize(
    "keys,table",
    [
        (
            ["k1"],
            pa.table(
                {
                    "k1": ["x", "x", "y", "y", "z"],
                    "d1": [1, 2, 0, 4, 1],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
        (
            ["k1", "k2"],
            pa.table(
                {
                    "k1": ["x", "y", "y", "y", "x"],
                    "d1": [1, 2, 0, 4, 1],
                    "k2": ["y", "x", "y", "x", "y"],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        [("d1", "sum", "sum")],
        [("d1", "min", "min"), ("d1", "max", "max")],
        [("d2", "mean", "mean"), ("d1", "product", "prod")],
    ],
)
def test_aggregation_basic(table, keys, aggs):
    expect = arrow_groupby(table, keys, aggs)

    tbl = LogicalTable.from_arrow(table)
    result = groupby_aggregation(tbl, keys, aggs)

    # sort before testing as the order of keys is arbitrary
    sort_keys = [(key, "ascending") for key in keys]
    assert_arrow_table_equal(
        result.to_arrow().sort_by(sort_keys), expect.sort_by(sort_keys), True
    )


@pytest.mark.parametrize("value_type", ["int64", "float64", "uint8"])
@pytest.mark.parametrize("key_type", ["int64", "string", "float32"])
@pytest.mark.parametrize(
    "aggregation",
    [
        "sum",
        "product",
        "min",
        "max",
        "count",
        "count_all",
        "mean",
        "variance",
        "stddev",
        "approximate_median",
        "count_distinct",
    ],
)
def test_numeric_aggregations(value_type, key_type, aggregation):
    rng = np.random.RandomState(42)
    n_keys = 10
    n = 1000
    if key_type == "int64":
        key_a = pa.array(rng.randint(0, n_keys, n), type=pa.int64())
    elif key_type == "string":
        key_a = pa.array([f"key_{i % n_keys}" for i in range(n)], type=pa.string())
    elif key_type == "float32":
        key_a = pa.array(
            rng.randint(0, n_keys, n).astype(np.float32), type=pa.float32()
        )
    key_b = pa.array(rng.randint(0, n_keys, n), type=pa.int64())
    value_a = pa.array(rng.random(n).astype(value_type), type=value_type)
    value_b = pa.array(rng.random(n), type=pa.float64())

    table = pa.Table.from_arrays(
        [key_a, value_a, key_b, value_b], names=["key_a", "value_a", "key_b", "value_b"]
    )
    keys = ["key_a", "key_b"]
    column_aggregations = [
        (
            "value_a" if aggregation != "count_all" else None,
            aggregation,
            f"value_a_{aggregation}",
        ),
        (
            "value_b" if aggregation != "count_all" else None,
            aggregation,
            f"value_b_{aggregation}",
        ),
    ]

    expected = arrow_groupby(table, keys, column_aggregations)
    tbl = LogicalTable.from_arrow(table)
    result = groupby_aggregation(tbl, keys, column_aggregations)
    sort_keys = [(key, "ascending") for key in keys]
    assert_arrow_table_equal(
        result.to_arrow().sort_by(sort_keys), expected.sort_by(sort_keys), True
    )


@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        # "product",
        # TODO: min/max require mask_nan's.
        # "min",
        # "max",
        "mean",
        "count",
        "len",  # -> "count_all"
        # TODO(seberg): Need to enable any/all in general
        # (this is slightly harder due to NULL logic handling)
        # "any",
        # "all",
        # "var",  # -> "variance",
        # "std",  # -> "stddev",
        # "approximate_median",
        # "n_unique",  # -> "count_distinct"
        # "tdigest",
    ],
)
def test_polars_basic(agg):
    pl = pytest.importorskip("polars")

    mask = np.random.randint(2, size=10_000, dtype=bool)
    q = pl.DataFrame(
        {
            "a": pa.array(np.random.random(10_000), mask=mask),
            "b": np.random.randint(100, size=10_000),
        }
    ).lazy()

    if agg == "len":
        q1 = q.group_by("a").agg(pl.len())
        q2 = q.group_by("b").agg(pl.len())
    elif agg not in {"any", "all"}:
        q1 = q.group_by("a").agg(getattr(pl.col("b"), agg)())
        q2 = q.group_by("b").agg(getattr(pl, agg)("a"))
    else:
        q1 = q.cast({"b": pl.Boolean}).group_by("a").agg(getattr(pl.col("b"), agg)())
        q2 = q.cast({"a": pl.Boolean}).group_by("b").agg(getattr(pl.col("a"), agg)())

    assert_matches_polars(q1.sort("a"), approx=True)
    assert_matches_polars(q2.sort("b"), approx=True)
