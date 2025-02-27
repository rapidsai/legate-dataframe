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
import legate.core
import pytest
from cudf._lib.binaryop import binaryop as cudf_binaryop

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.binaryop import binary_operation, binary_operator
from legate_dataframe.testing import assert_frame_equal, get_column_set


def gen_random_series(nelem: int, num_nans: int) -> cudf.Series:
    a = cupy.random.random(nelem)
    a[cupy.random.choice(a.size, num_nans, replace=False)] = cupy.nan
    return cudf.Series(a, nan_as_null=True)


@pytest.mark.parametrize(
    "op",
    [
        binary_operator.ADD,
        binary_operator.SUB,
        binary_operator.MUL,
        binary_operator.DIV,
        binary_operator.MOD,
        binary_operator.POW,
    ],
)
def test_arithmetic_operations(op: binary_operator):
    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    lg_a = LogicalColumn.from_cudf(a._column)
    lg_b = LogicalColumn.from_cudf(b._column)

    res = binary_operation(lg_a, lg_b, op, a.dtype)
    expect = getattr(a, op.name.lower())(b)

    assert_frame_equal(res, expect, default_column_name="col0")


@pytest.mark.parametrize(
    "op",
    [
        binary_operator.ADD,
        binary_operator.SUB,
        binary_operator.MUL,
        binary_operator.DIV,
        binary_operator.POW,
    ],
)
@pytest.mark.parametrize("cudf_column", get_column_set(["int32", "float32", "int64"]))
@pytest.mark.parametrize(
    "scalar",
    [
        cudf.Scalar(None, dtype="int32"),
        cudf.Scalar(42, dtype="int64"),
        cudf.Scalar(42, dtype="float64"),
        cudf.Scalar(-42, dtype="float32"),
        42,
        -42.0,
        legate.core.Scalar(42, dtype=legate.core.types.int64),
        legate.core.Scalar(-42, dtype=legate.core.types.float16),
    ],
)
def test_scalar_input(cudf_column, op, scalar):
    op_str = f"__{op.name.lower()}__"
    col = LogicalColumn.from_cudf(cudf_column)
    if isinstance(scalar, legate.core.Scalar):
        cudf_scalar = scalar.value()  # cudf doesn't understand legate's scalars
    else:
        cudf_scalar = scalar

    res = binary_operation(col, scalar, op, cudf_column.dtype)
    expect = cudf_binaryop(cudf_column, cudf_scalar, op_str, cudf_column.dtype)
    assert_frame_equal(res, expect)

    res = binary_operation(scalar, col, op, cudf_column.dtype)
    expect = cudf_binaryop(cudf_scalar, cudf_column, op_str, cudf_column.dtype)
    assert_frame_equal(res, expect)

    result = binary_operation(scalar, scalar, op, cudf_column.dtype)
    assert result.scalar()  # if both inputs are scalar, the result is also
