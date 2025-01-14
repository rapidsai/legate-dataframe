# Copyright (c) 2025, NVIDIA CORPORATION
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

from legate_dataframe import LogicalTable
from legate_dataframe.lib.sort import sort
from legate_dataframe.testing import assert_frame_equal


@pytest.mark.parametrize(
    "values",
    [cupy.arange(0, 1000), cupy.arange(0, -1000, -1), cupy.ones(1000), cupy.ones(1)],
)
def test_basic(values):
    df = cudf.DataFrame({"a": values})

    lg_df = LogicalTable.from_cudf(df)
    lg_sorted = sort(lg_df, ["a"])

    df_sorted = df.sort_values(by=["a"])

    assert_frame_equal(lg_sorted, df_sorted)


@pytest.mark.parametrize(
    "values,stable",
    [
        (cupy.arange(0, 1000), False),
        (cupy.arange(0, 1000), True),
        (cupy.arange(0, -1000, -1), False),
        (cupy.arange(0, -1000, -1), True),
        (cupy.ones(1000), True),
        (cupy.ones(3), True),
    ],
)
def test_basic_with_extra_column(values, stable):
    # Similar as above, but additional column should stay shuffle same.
    df = cudf.DataFrame({"a": values, "b": cupy.arange(len(values))})

    lg_df = LogicalTable.from_cudf(df)
    lg_sorted = sort(lg_df, ["a"], stable=stable)

    if not stable:
        df_sorted = df.sort_values(by=["a"])
    else:
        df_sorted = df.sort_values(by=["a"], kind="stable")

    assert_frame_equal(lg_sorted, df_sorted)


@pytest.mark.parametrize("reversed", [True, False])
def test_shifted_equal_window(reversed):
    # The tricky part abort sorting are the exact splits for exchanging.
    # assume we have at least two gpus/workders.  Shift a window of 50
    # (i.e. half of each worker), through, to see if it gets split incorrectly.
    for i in range(150):
        before = cupy.arange(i)
        constant = cupy.full(50, i)
        after = cupy.arange(50 + i, 200)
        values = cupy.concatenate([before, constant, after])
        if reversed:
            values = values[::-1].copy()

        # Need a second column to check the splits:
        df = cudf.DataFrame({"a": values, "b": cupy.arange(200)})

        lg_df = LogicalTable.from_cudf(df)
        lg_sorted = sort(lg_df, ["a"], stable=True)
        df_sorted = df.sort_values(by=["a"], kind="stable")

        assert_frame_equal(lg_sorted, df_sorted)
