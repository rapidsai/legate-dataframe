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

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.copying import copy_if_else
from legate_dataframe.testing import (
    assert_matches_polars,
    gen_random_series,
    get_pyarrow_column_set,
)


@pytest.mark.parametrize(
    "array", get_pyarrow_column_set(["int32", "float32", "int64", "bool"])
)
def test_copy_if_else(array):
    cond = pa.array(np.random.randint(0, 2, size=len(array)).astype(np.bool_))
    cond_lg = LogicalColumn.from_arrow(cond)
    lhs = LogicalColumn.from_arrow(array)
    rhs = LogicalColumn.from_arrow(array[::-1])  # reverse to make it interesting!

    res = copy_if_else(cond_lg, lhs, rhs)

    expect = pa.compute.if_else(cond, array, array[::-1])
    assert expect == res.to_arrow()


def test_copy_if_else_string():
    array = pa.array(["a", "b", "c", "this is a longer string!"])
    cond = pa.array([False, True, True, False])
    cond_lg = LogicalColumn.from_arrow(cond)
    lhs = LogicalColumn.from_arrow(array)
    rhs = LogicalColumn.from_arrow(array[::-1])  # reverse to make it interesting!

    res = copy_if_else(cond_lg, lhs, rhs)

    expect = pa.compute.if_else(cond, array, array[::-1])
    assert expect == res.to_arrow()


@pytest.mark.parametrize(
    "lhs,rhs,cond",
    [
        (pa.array([1, 2, 3]), pa.scalar(42), pa.array([True, False, True])),
        (pa.scalar(42), pa.array([1, 2, 3]), pa.array([True, False, True])),
        (pa.scalar(42), pa.scalar(1), pa.array([True, False, True])),
        (pa.scalar(42), pa.scalar(1), pa.scalar(True)),
    ],
)
def test_copy_if_else_scalar(lhs, rhs, cond):
    lhs_lg = LogicalColumn.from_arrow(lhs)
    rhs_lg = LogicalColumn.from_arrow(rhs)
    cond_lg = LogicalColumn.from_arrow(cond)

    res = copy_if_else(cond_lg, lhs_lg, rhs_lg)
    expect = pa.compute.if_else(cond, lhs, rhs)

    if lhs_lg.is_scalar() and rhs_lg.is_scalar() and cond_lg.is_scalar():
        assert res.is_scalar()
        assert res.to_arrow() == pa.array([expect])
    else:
        assert not res.is_scalar()
        assert res.to_arrow() == expect


def test_copy_if_else_errors():
    col1 = LogicalColumn.from_arrow(pa.array([1, 2, 3]))
    col2 = LogicalColumn.from_arrow(pa.array([1.0, 2.0, 3.0]))
    cond = LogicalColumn.from_arrow(pa.array([True, False, True]))
    with pytest.raises(ValueError):
        copy_if_else(cond, col1, col2)  # different type for col1 and col2

    with pytest.raises(ValueError):
        copy_if_else(col1, col1, col1)  # cond is not a boolean column


def test_polars_ternary_copy_if_else():
    pl = pytest.importorskip("polars")

    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    c = pa.array(np.random.randint(0, 2, size=1000).astype(np.bool_))

    a_s = pl.from_arrow(a)
    b_s = pl.from_arrow(b)
    c_s = pl.from_arrow(c)

    q = pl.LazyFrame({"a": a_s, "b": b_s, "c": c_s}).with_columns(
        result=pl.when("c").then(pl.col("a")).otherwise(pl.col("b"))
    )
    assert_matches_polars(q)

    # Also check with at least one scalar involved:
    q = pl.LazyFrame({"a": a_s, "b": b_s, "c": c_s}).with_columns(
        result=pl.when("c").then(pl.col("a")).otherwise(42)
    )
    assert_matches_polars(q)
