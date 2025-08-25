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

import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.timestamps import extract_timestamp_component, to_timestamps
from legate_dataframe.testing import assert_matches_polars


@pytest.mark.parametrize(
    "col",
    [pa.array(["2010-06-19T13:15", "2011-06-19T13:25", "2010-07-19T13:35"])],
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
def test_to_timestamps(col, timestamp_type):
    lg_col = LogicalColumn.from_arrow(col)
    format = "%Y-%m-%dT%H:%M"
    res = to_timestamps(lg_col, timestamp_type, format)
    expect = pa.compute.strptime(
        col,
        format,
        unit=timestamp_type[timestamp_type.find("[") + 1 : timestamp_type.find("]")],
    )
    assert res.to_arrow() == expect


@pytest.mark.parametrize(
    "timestamp_type",
    [
        "s",
        "ms",
        "us",
        "ns",
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ],
)
def test_extract_timestamp_component(timestamp_type, field):
    from datetime import datetime

    col = pa.array(
        [
            pa.scalar(
                datetime(2005, 6, 1, 13, 33),
                type=pa.timestamp(timestamp_type, tz="UTC"),
            )
        ],
    )
    lg_col = LogicalColumn.from_arrow(col)
    res = extract_timestamp_component(lg_col, field)
    if field == "day_of_week":
        expected = pa.compute.day_of_week(col, count_from_zero=False)
    else:
        expected = pa.compute.call_function(field, [col])
    assert res.to_arrow() == expected


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
