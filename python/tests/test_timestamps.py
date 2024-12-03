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
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.timestamps import (
    DatetimeComponent,
    extract_timestamp_component,
    to_timestamps,
)
from legate_dataframe.testing import assert_frame_equal


@pytest.mark.parametrize(
    "df",
    [
        cudf.DataFrame(
            {"a": ["2010-06-19T13:15", "2011-06-19T13:25", "2010-07-19T13:35"]}
        )
    ],
)
@pytest.mark.parametrize(
    "timestamp_type",
    [
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_to_timestamps(df, timestamp_type):
    expect = cudf.to_datetime(df["a"]).astype(timestamp_type)
    lg_col = LogicalColumn.from_cudf(df._columns[0])
    res = to_timestamps(lg_col, timestamp_type, "%Y-%m-%dT%H:%M:%SZ")

    assert_frame_equal(res, expect, default_column_name="a")


@pytest.mark.parametrize(
    "timestamp_type",
    [
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        DatetimeComponent.year,
        DatetimeComponent.month,
        DatetimeComponent.day,
        DatetimeComponent.weekday,
        DatetimeComponent.hour,
        DatetimeComponent.minute,
        DatetimeComponent.second,
        # Cudf/pandas microsecond includes the milliseconds (so doesn't exist):
        # DatetimeComponent.millisecond_fraction,
        DatetimeComponent.microsecond_fraction,
        DatetimeComponent.nanosecond_fraction,
        DatetimeComponent.day_of_year,
    ],
)
def test_extract_timestamp_component(timestamp_type, field):
    col = cudf.Series(
        ["2010-06-19T13:15:12.1232634", "2011-06-20T13:25:11.2789543"]
    ).astype(timestamp_type)
    expected = getattr(col.dt, field.name.removesuffix("_fraction"))
    if field == DatetimeComponent.weekday:
        # cudf subtracts 1 and that seems to cast:
        expected += 1
        expected = expected.astype("int16")
    elif field == DatetimeComponent.microsecond_fraction:
        # Remove milliseconds from cudf result and cast to int16:
        expected = expected % 1000
        expected = expected.astype("int16")

    lg_col = LogicalColumn.from_cudf(col._column)
    res = extract_timestamp_component(lg_col, field)

    assert_frame_equal(res, expected)
