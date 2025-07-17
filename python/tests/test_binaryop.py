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

import operator

import legate.core
import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.binaryop import binary_operation
from legate_dataframe.testing import (
    assert_matches_polars,
    gen_random_series,
    get_pyarrow_column_set,
)

ops = ["add", "subtract", "multiply"]
ops_logical = [
    "and",
    "or",
    "and_kleene",
    "or_kleene",
]
ops_comparison = [
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
]


@pytest.mark.parametrize("op", ops)
def test_arithmetic_operations(op: str):
    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    lg_a = LogicalColumn.from_arrow(a)
    lg_b = LogicalColumn.from_arrow(b)

    res = binary_operation(lg_a, lg_b, op, np.float64)
    expect = pa.compute.call_function(op, [a, b])
    assert expect == res.to_arrow()


@pytest.mark.parametrize("op", ops_logical + ops_comparison)
def test_bool_out_operations(op: str):
    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    lg_a = LogicalColumn.from_arrow(a)
    lg_b = LogicalColumn.from_arrow(b)

    res = binary_operation(lg_a, lg_b, op, "bool")
    if op in ops_logical:
        expect = pa.compute.call_function(op, [a.cast("bool"), b.cast("bool")])
    else:
        expect = pa.compute.call_function(op, [a, b])
    assert expect == res.to_arrow()


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("array", get_pyarrow_column_set(["int32", "float32", "int64"]))
@pytest.mark.parametrize(
    "scalar",
    [
        pa.scalar(None, type="int32"),
        pa.scalar(42, type="int64"),
        pa.scalar(42, type="float64"),
        pa.scalar(-42, type="float32"),
        42,
        -42.0,
        legate.core.Scalar(42, dtype=legate.core.types.int64),
        legate.core.Scalar(-42, dtype=legate.core.types.float16),
    ],
)
def test_scalar_input(array, op, scalar):
    col = LogicalColumn.from_arrow(array)

    res = binary_operation(col, scalar, op, array.type)

    pa_scalar = (
        pa.scalar(scalar.value())
        if isinstance(scalar, legate.core.Scalar)
        else pa.scalar(scalar)
    )
    expect = pa.compute.call_function(op, [array, pa_scalar]).cast(array.type)
    assert expect == res.to_arrow()

    res = binary_operation(scalar, col, op, array.type)
    expect = pa.compute.call_function(op, [pa_scalar, array]).cast(array.type)
    assert expect == res.to_arrow()

    result = binary_operation(scalar, scalar, op, array.type)
    assert result.is_scalar()  # if both inputs are scalar, the result is also


operators = ["add", "sub", "mul", "and_", "or_", "eq", "ne", "lt", "le", "gt", "ge"]


@pytest.mark.parametrize("op", operators)
def test_binary_operation_polars(op):
    pl = pytest.importorskip("polars")

    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    if op in {"and_", "or_"}:
        # and_ and or_ are bitwise, and require bool inputs.
        a = a.cast("bool")
        b = b.cast("bool")

    a_s = pl.from_arrow(a)
    b_s = pl.from_arrow(b)

    # Need to work with a lazyframe, as there is no lazy series.
    q = pl.LazyFrame({"a": a_s, "b": b_s}).with_columns(
        a_b=getattr(operator, op)(pl.col("a"), pl.col("b"))
    )
    assert_matches_polars(q)


@pytest.mark.parametrize("mode", ["none", "left", "right", "both"])
def test_between_polars(mode):
    pl = pytest.importorskip("polars")

    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    c = gen_random_series(nelem=1000, num_nans=10)

    a_s = pl.from_arrow(a)
    b_s = pl.from_arrow(b)
    c_s = pl.from_arrow(c)

    q = pl.LazyFrame({"a": a_s, "b": b_s, "c": c_s}).with_columns(
        pl.col("a").is_between(pl.col("b"), pl.col("c"), closed=mode)
    )
    assert_matches_polars(q)
