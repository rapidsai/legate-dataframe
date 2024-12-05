# Copyright 2024 NVIDIA Corporation
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
import legate.core.types as lg_type
from legate.core import get_legate_runtime
from pylibcudf.unary import UnaryOperator

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.unaryop import unary_operation
from legate_dataframe.testing import assert_frame_equal


def test_python_launched_tasks():
    col = LogicalColumn.from_cudf(cudf.Series(cupy.random.random(100))._column)

    # Launch an unary task using the Cython API
    expect = unary_operation(col, UnaryOperator.ABS)

    # Launch an unary task using the Python API

    # First, we need to find the library, which was registered implicitly
    # by the `unary_operation()` call above.
    runtime = get_legate_runtime()
    lib = runtime.find_library("legate_dataframe")

    # Then, we can create the task and provide the task arguments using the
    # exact same order as in the task implementation ("unaryop.cpp").
    task = runtime.create_auto_task(lib, 5)  # TODO: get the enum of `UnaryOperator`
    task.add_scalar_arg(UnaryOperator.ABS.value, dtype=lg_type.int32)
    col.add_as_next_task_input(task)
    result = LogicalColumn.empty_like_logical_column(col)
    result.add_as_next_task_output(task)
    task.execute()
    assert_frame_equal(result, expect)
