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
import pytest
from pylibcudf.unary import UnaryOperator

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.unaryop import cast, unary_operation
from legate_dataframe.testing import assert_frame_equal


@pytest.mark.parametrize("op", UnaryOperator)
def test_unary_operation(op):
    if op in (UnaryOperator.BIT_INVERT, UnaryOperator.NOT):
        series = cudf.Series(cupy.random.randint(0, 2, size=1000).astype(bool))
    else:
        series = cudf.Series(cupy.random.random(1000))
    col = LogicalColumn.from_cudf(series._column)
    res = unary_operation(col, op)
    expect = series._column.unary_operator(op.name)
    assert_frame_equal(res, expect)


def test_unary_operation_scalar():
    # It makes sense for unary operators to propagte "scalar" information
    # check that.
    scalar = cudf.Scalar(-3).device_value

    scalar_col = LogicalColumn.from_cudf(scalar)
    res = unary_operation(scalar_col, UnaryOperator.ABS)

    assert res.is_scalar()
    assert res.to_cudf_scalar().value == 3


@pytest.mark.parametrize("from_dtype", ["int8", "uint64", "float32", "float64"])
@pytest.mark.parametrize("to_dtype", ["int8", "uint64", "float32", "float64"])
def test_cast(from_dtype, to_dtype):
    arr = cupy.random.randint(0, 1000, size=1000).astype(from_dtype)
    series = cudf.Series(arr)

    expected = series.astype(to_dtype)
    col = LogicalColumn.from_cudf(series._column)
    res = cast(col, to_dtype)
    assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "from_series,to_dtype",
    [
        (cudf.Series([1234, 1234534], dtype="m8[s]"), "uint64"),
        (cudf.Series([1234, 1234534], dtype="m8[s]"), "float64"),
    ],
)
def test_cast_timedelta(from_series, to_dtype):
    # Test timedelta to numeric cast, libcudf doesn't cast datetimes directly
    col = LogicalColumn.from_cudf(from_series._column)
    res = cast(col, to_dtype)
    assert_frame_equal(res, from_series.astype(to_dtype))


def test_bad_cast():
    # We try to reject invalid casts (before the Task would crash hard).
    # Unfortunately, libcudf fails to reject some invalid cases :(.
    col = LogicalColumn.from_cudf(cudf.Series([1, 2, 3])._column)
    with pytest.raises(ValueError, match="Cannot cast column to specified type"):
        cast(col, "str")


def test_cast_scalar():
    # It makes sense for unary operators to propagte "scalar" information
    # check that.
    scalar = cudf.Scalar(-3).device_value

    scalar_col = LogicalColumn.from_cudf(scalar)
    res = cast(scalar_col, "int8")

    assert res.is_scalar()
    assert res.to_cudf_scalar().value == -3
