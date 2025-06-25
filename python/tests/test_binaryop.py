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

import legate.core
import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.binaryop import binary_operation
from legate_dataframe.testing import get_pyarrow_column_set


def gen_random_series(nelem: int, num_nans: int) -> pa.Array:
    rng = np.random.default_rng(42)
    a = rng.random(nelem)
    nans = np.zeros(nelem, dtype=bool)
    nans[rng.choice(a.size, num_nans, replace=False)] = True
    return pa.array(a, mask=nans)


ops = ["add", "subtract", "multiply"]


@pytest.mark.parametrize("op", ops)
def test_arithmetic_operations(op: str):
    a = gen_random_series(nelem=1000, num_nans=10)
    b = gen_random_series(nelem=1000, num_nans=10)
    lg_a = LogicalColumn.from_arrow(a)
    lg_b = LogicalColumn.from_arrow(b)

    res = binary_operation(lg_a, lg_b, op, np.float64)
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
