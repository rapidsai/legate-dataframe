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

import cudf
import cupy
import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.unaryop import cast, unary_operation
from legate_dataframe.testing import assert_frame_equal, gen_random_series

ops = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "ln",
    "sqrt",
    "ceil",
    "floor",
    "abs",
    "round",
    "invert",
    "negate",
]


@pytest.mark.parametrize("op", ops)
def test_unary_operation(op):
    if op == "invert":
        array = pa.array(np.random.randint(0, 2, size=1000).astype(bool))
    else:
        array = pa.array(np.random.random(1000))
    col = LogicalColumn.from_arrow(array)
    res = unary_operation(col, op)
    expect = pa.compute.call_function(op, [array])
    assert np.allclose(
        expect.to_numpy(zero_copy_only=False),
        res.to_arrow().to_numpy(zero_copy_only=False),
        equal_nan=True,
    )


def test_unary_operation_scalar():
    # It makes sense for unary operators to propagte "scalar" information
    # check that.
    scalar = pa.scalar(-42.0, type="float64")

    scalar_col = LogicalColumn.from_arrow(scalar)
    res = unary_operation(scalar_col, "abs")

    assert res.is_scalar()
    assert res.to_arrow()[0].as_py() == 42.0


@pytest.mark.parametrize("from_dtype", ["int8", "uint64", "float32", "float64"])
@pytest.mark.parametrize("to_dtype", ["int8", "uint64", "float32", "float64"])
def test_cast(from_dtype, to_dtype):
    arr = cupy.random.randint(0, 1000, size=1000).astype(from_dtype)
    series = cudf.Series(arr)

    expected = series.astype(to_dtype)
    col = LogicalColumn.from_cudf(series._column)
    res = cast(col, to_dtype)
    assert_frame_equal(res, expected)


def test_cast_timedelta():
    # Test timedelta to numeric cast
    # arrow supports casting to int64
    to_dtype = "int64"
    array = pa.array([1234, 1234534], type=pa.duration("s"))
    col = LogicalColumn.from_arrow(array)
    res = cast(col, to_dtype)
    assert res.to_arrow() == array.cast(to_dtype)


def test_cast_scalar():
    # It makes sense for unary operators to propagte "scalar" information
    # check that.
    scalar = pa.scalar(-3)

    scalar_col = LogicalColumn.from_arrow(scalar)
    res = cast(scalar_col, "int8")

    assert res.is_scalar()
    assert res.to_arrow()[0].as_py() == -3


polars_ops = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "sqrt",
    # "cbrt",
    "ceil",
    "floor",
    "abs",
    # bit_invert, "not" not attributes, but also not on floats
]


@pytest.mark.parametrize("op", polars_ops)
def test_unary_operation_polars(op):
    pl = pytest.importorskip("polars")
    import legate_dataframe.ldf_polars  # noqa: F401

    a = gen_random_series(nelem=1000, num_nans=10)
    q = pl.LazyFrame({"a": a}).with_columns(res=getattr(pl.col("a"), op)())

    # Need to use approximate equality here
    res_pl = np.asarray(q.collect().to_arrow())
    res_ldf = np.asarray(q.legate.collect().to_arrow())

    np.testing.assert_array_almost_equal(res_pl, res_ldf)
