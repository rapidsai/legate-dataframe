# Copyright (c) 2023-2024, NVIDIA CORPORATION
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
from legate_dataframe.lib.unaryop import unary_operation
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
