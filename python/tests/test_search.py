# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.search import contains
from legate_dataframe.testing import assert_matches_polars, gen_random_series


@pytest.mark.parametrize(
    "haystack,needles,expected",
    [
        ([1, 2, 3], [2, 3, 4], [True, True, False]),
        (["a", "b"], ["b", "a", "c"], [True, True, False]),
        # Check null handling (haystack nulls ignored, needles propagated)
        (
            pa.array([1, 2, 3], mask=[False, True, False]),
            pa.array([2, 3, 1], mask=[False, False, True]),
            pa.array([False, True, True], mask=[False, False, True]),
        ),
    ],
)
def test_contains(haystack, needles, expected):
    haystack = pa.array(haystack)
    needles = pa.array(needles)
    haystack_lg = LogicalColumn.from_arrow(haystack)
    needles_lg = LogicalColumn.from_arrow(needles)

    result = contains(haystack_lg, needles_lg)
    assert result.to_arrow() == pa.array(expected)


def test_contains_errors():
    col1 = LogicalColumn.from_arrow(pa.array([1, 2, 3]))
    col2 = LogicalColumn.from_arrow(pa.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        contains(col1, col2)  # different types


def test_polars_contains():
    pl = pytest.importorskip("polars")

    arr = gen_random_series(nelem=1000, num_nans=200)
    haystack = pl.LazyFrame({"a": pl.from_arrow(arr[:200])})
    needles = pl.Series(pl.from_arrow(arr))

    # Compare with polars
    q = haystack.with_columns(contained=pl.col("a").is_in(needles))
    q.collect()
    assert_matches_polars(q)
