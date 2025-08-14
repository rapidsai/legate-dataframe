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
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.timestamps import (
    DatetimeComponent,
    extract_timestamp_component,
    to_timestamps,
)
from legate_dataframe.testing import assert_frame_equal, assert_matches_polars


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
        DatetimeComponent.YEAR,
        DatetimeComponent.MONTH,
        DatetimeComponent.DAY,
        DatetimeComponent.WEEKDAY,
        DatetimeComponent.HOUR,
        DatetimeComponent.MINUTE,
        DatetimeComponent.SECOND,
        DatetimeComponent.MILLISECOND,
        DatetimeComponent.MICROSECOND,
        DatetimeComponent.NANOSECOND,
    ],
)
def test_extract_timestamp_component(timestamp_type, field):
    col = cudf.Series(
        ["2010-06-19T13:15:12.1232634", "2011-06-20T13:25:11.2789543"]
    ).astype(timestamp_type)

    if field == DatetimeComponent.MILLISECOND:
        # millisecond are not exposed directly but included in microseconds:
        expected = col.dt.microsecond // 1000
        expected = expected.astype("int16")
    elif field == DatetimeComponent.WEEKDAY:
        # cudf subtracts 1 and that seems to cast:
        expected = col.dt.weekday + 1
        expected = expected.astype("int16")
    elif field == DatetimeComponent.MICROSECOND:
        # Remove milliseconds from cudf result:
        expected = col.dt.microsecond % 1000
        expected = expected.astype("int16")
    else:
        expected = getattr(col.dt, field.name.lower())

    lg_col = LogicalColumn.from_cudf(col._column)
    res = extract_timestamp_component(lg_col, field)

    assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    "get_dtype",
    [
        # lambda pl: pl.Date(),  complicated with arrow due to 32bit
        lambda pl: pl.Datetime(time_unit="ms"),
        lambda pl: pl.Datetime(time_unit="us"),
        lambda pl: pl.Datetime(time_unit="ns"),
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "microsecond",
        "nanosecond",
        "year",
        "month",
        "day",
        "weekday",
        "hour",
        "minute",
        "second",
        "millisecond",
    ],
)
def test_extract_timestamp_component_polars(get_dtype, field):
    pl = pytest.importorskip("polars")

    df = pl.DataFrame(
        [
            pl.Series(
                "a",
                ["2010-06-19T13:15:12.1232634", "2011-06-20T13:25:11.2789543"],
            )
            .str.to_datetime()
            .cast(get_dtype(pl))
        ]
    )

    q = df.lazy().with_columns(getattr(pl.col("a").dt, field)())

    assert_matches_polars(q, allow_exceptions=pl.exceptions.InvalidOperationError)
