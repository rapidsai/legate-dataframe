# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib import strings
from legate_dataframe.testing import assert_matches_polars

match_funcs = ["starts_with", "ends_with", "match_substring", "match_substring_regex"]


@pytest.mark.parametrize("match_func", match_funcs)
@pytest.mark.parametrize("pattern", ["", "hel", "o", "o.+d"])  # codespell:ignore
def test_matches(match_func, pattern):
    arr = pa.array(
        ["hello", "olhel", "world", "wo.+d"] * 2,  # codespell:ignore
        mask=[True, True, True, True, False, False, False, False],
    )
    column = LogicalColumn.from_arrow(arr)
    result = strings.match(match_func, column, pattern)
    expected = getattr(pa.compute, match_func)(arr, pattern)
    assert expected == result.to_arrow()


@pytest.mark.parametrize("match_func", match_funcs)
def test_matches_empty(match_func):
    column = LogicalColumn.from_arrow(pa.array([], "string"))
    result = strings.match(match_func, column, "o")
    assert result.to_arrow() == pa.array([], "bool")


def test_bad_function():
    column = LogicalColumn.from_arrow(pa.array(["hello", "world", "help", "hero"]))
    with pytest.raises(ValueError):
        strings.match("bad_function", column, "o")


@pytest.mark.parametrize(
    "pl_func",
    [
        lambda x, p: getattr(x.str, "starts_with")(p),
        lambda x, p: getattr(x.str, "ends_with")(p),
        lambda x, p: getattr(x.str, "contains")(p),
        lambda x, p: getattr(x.str, "contains")(p, literal=True),
    ],
)
@pytest.mark.parametrize("pattern", ["", "hel", "o", "o.+d", "l$"])  # codespell:ignore
def test_polars_matches(pl_func, pattern):
    pl = pytest.importorskip("polars")

    arr = pa.array(
        ["hello", "olhel", "world", "wo.+d"] * 2,  # codespell:ignore
        mask=[True, True, True, True, False, False, False, False],
    )

    q = pl.LazyFrame({"a": arr})
    q = q.with_columns(matches=pl_func(pl.col("a"), pattern))
    assert_matches_polars(q)
