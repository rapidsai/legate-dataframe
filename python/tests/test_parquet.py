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

import glob

import legate.core
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from legate.core import get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.parquet import parquet_read, parquet_read_array, parquet_write
from legate_dataframe.lib.replace import replace_nulls
from legate_dataframe.testing import (
    assert_arrow_table_equal,
    assert_frame_equal,
    assert_matches_polars,
    std_dataframe_set_cpu,
)


def write_partitioned_parquet(table, path, npartitions=1):
    if table.num_rows == 0:
        pq.write_table(table, f"{path}/part-0.parquet")
    partition_size = table.num_rows // npartitions
    for i in range(npartitions):
        start = i * partition_size
        end = min((i + 1) * partition_size, table.num_rows)
        if i == npartitions - 1:
            # Last partition may be larger if not evenly divisible
            end = table.num_rows
        if start >= end:
            break
        partition = table[start:end]
        pq.write_table(partition, f"{path}/part-{i}.parquet")


@pytest.mark.parametrize("df", std_dataframe_set_cpu())
def test_write(tmp_path, df):
    tbl = LogicalTable.from_arrow(df)

    parquet_write(tbl, path=tmp_path)
    get_legate_runtime().issue_execution_fence(block=True)

    # Read the files back with pyarrow then compare with the original
    files = sorted(glob.glob(str(tmp_path) + "/*.parquet"))
    tables = [pq.read_table(file) for file in files]

    # Concatenate the tables
    combined_table = pa.concat_tables(tables)

    assert_arrow_table_equal(combined_table, df)


@pytest.mark.parametrize("columns", [None, ["b"], ["a", "b"], ["b", "a"], []])
@pytest.mark.parametrize("df", std_dataframe_set_cpu())
@pytest.mark.parametrize("ignore_row_groups", [True, False])
def test_read(tmp_path, df, columns, ignore_row_groups, glob_string="/*"):
    pq.write_table(df, str(tmp_path) + "/test.parquet")

    has_cols = columns is None or all(c in df.column_names for c in columns)
    if not has_cols:
        with pytest.raises(ValueError):
            parquet_read(
                str(tmp_path) + glob_string,
                columns=columns,
                ignore_row_groups=ignore_row_groups,
            )
    else:
        tbl = parquet_read(
            str(tmp_path) + glob_string,
            columns=columns,
            ignore_row_groups=ignore_row_groups,
        )
        if columns is not None:
            df = df.select(columns)

        if tbl.get_column_names():
            assert_arrow_table_equal(tbl.to_arrow(), df)
        else:
            # Table is empty (and has no rows). cudf has an index, though.
            assert len(df.column_names) == 0


@pytest.mark.parametrize("ignore_row_groups", [True, False])
def test_read_single_rows(tmp_path, ignore_row_groups, glob_string="/*"):
    df = pa.table({"a": np.arange(1, dtype="int64")})
    pq.write_table(df, str(tmp_path) + "/test.parquet")
    tbl = parquet_read(str(tmp_path) + glob_string, ignore_row_groups=ignore_row_groups)
    assert_arrow_table_equal(tbl.to_arrow(), df)


@pytest.mark.parametrize("ignore_row_groups", [True, False])
def test_read_many_files_per_rank(tmp_path, ignore_row_groups, glob_string="/*"):
    # Use uneven number to test splitting
    df = pa.table({"a": np.arange(983, dtype="int64")})
    npartitions = 100
    write_partitioned_parquet(df, tmp_path, npartitions=npartitions)
    assert len(glob.glob(str(tmp_path) + glob_string)) == npartitions
    tbl = parquet_read(str(tmp_path) + glob_string, ignore_row_groups=ignore_row_groups)

    # NOTE: Right now the C-code does not attempt to "natural" sort parquet
    #       files.  So more with more than 10 files the order of rows is not
    #       preserved at this time.
    assert_arrow_table_equal(tbl.to_arrow().sort_by("a"), df)


def test_read_array(tmp_path, npartitions=2, glob_string="/*"):
    cn = pytest.importorskip("cupynumeric")

    c = pa.array(
        np.arange(2, 10002, dtype="float32"), mask=np.array([True, False] * 5000)
    )
    # we have to explicitly tell pyarrow these arrays are non-nullable for some reason
    df = pa.table(
        {
            "a": pa.array(np.arange(10000, dtype="float32")),
            "b": np.arange(1, 10001, dtype="float32"),
            "c": c,  # only column with masked values
            "d": np.arange(3, 10003, dtype="float32"),
        },
        schema=pa.schema(
            [
                ("a", pa.float32(), False),
                ("b", pa.float32(), False),
                ("c", pa.float32(), True),  # nullable column
                ("d", pa.float32(), False),
            ]
        ),
    )

    write_partitioned_parquet(df, tmp_path, npartitions=npartitions)
    tbl = parquet_read(str(tmp_path) + glob_string, columns=["b", "a", "d", "c"])
    tbl = LogicalTable(
        [
            tbl["b"],
            tbl["a"],
            tbl["d"],
            replace_nulls(tbl["c"], pa.scalar(0.5, "float32")),
        ],
        ["b", "a", "d", "c"],
    )
    arr_from_tbl = tbl.to_array()

    # Need a null value to ensure there is no mask.
    null_value = legate.core.Scalar(0.5, legate.core.float32)
    arr = parquet_read_array(
        str(tmp_path) + glob_string, columns=["b", "a", "d", "c"], null_value=null_value
    )
    arr = cn.asarray(arr)

    assert arr.dtype == arr_from_tbl.dtype
    assert cn.array_equal(arr_from_tbl, arr)

    # Check if the mask behavior seems right for column "c"
    arr = parquet_read_array(str(tmp_path) + glob_string, columns=["c"])
    col_from_arr = LogicalColumn(arr.project(1, 0))
    col = parquet_read(str(tmp_path) + glob_string, columns=["c"])["c"]

    assert_frame_equal(col_from_arr, col)


def test_read_array_boolean(tmp_path):
    # booleans are a special case in arrow
    df = pa.table(
        {
            "a": np.resize([True, False], 1000),
        },
        schema=pa.schema(
            [("a", pa.bool_(), False)],
        ),
    )

    pq.write_table(df, str(tmp_path) + "/test.parquet")

    array = parquet_read_array(
        str(tmp_path) + "/*", null_value=legate.core.Scalar(False, legate.core.bool_)
    )
    assert (np.array(array).squeeze() == df.column("a").to_numpy()).all()


def test_read_array_cast(tmp_path, npartitions=2, glob_string="/*"):
    cn = pytest.importorskip("cupynumeric")

    df = pa.table(
        {
            "a": np.arange(10000, dtype="float32"),
            "b": np.arange(1, 10001, dtype="float64"),
        },
        schema=pa.schema([("a", pa.float32(), False), ("b", pa.float64(), False)]),
    )
    write_partitioned_parquet(df, tmp_path, npartitions=npartitions)

    null_value = legate.core.Scalar(0, legate.core.float32)
    arr = parquet_read_array(
        str(tmp_path) + glob_string,
        columns=["a", "b"],
        null_value=null_value,  # guarantee non-nullable result
        type=legate.core.float32,
    )
    arr = cn.asarray(arr)

    assert arr.dtype == "float32"
    assert cn.array_equal(arr[:, 0], cn.arange(10000, dtype="float32"))
    assert cn.array_equal(arr[:, 1], cn.arange(1, 10001, dtype="float32"))


@pytest.mark.parametrize("ignore_row_groups", [True, False])
def test_read_large(tmp_path, ignore_row_groups, npartitions=1, glob_string="/*"):
    pytest.importorskip("cupynumeric")

    # Create an array so large, that we should chunk (should be fine for testing)
    df = pa.table(
        {"a": np.ones(2**26, dtype="uint8")},
        schema=pa.schema([("a", pa.uint8(), False)]),
    )
    pq.write_table(df, str(tmp_path) + "/test.parquet")
    del df

    tbl = parquet_read(
        str(tmp_path) + glob_string, columns=["a"], ignore_row_groups=ignore_row_groups
    )
    assert tbl["a"].to_array().sum() == 2**26


def test_read_array_large(tmp_path, npartitions=1, glob_string="/*"):
    cn = pytest.importorskip("cupynumeric")

    # Create an array so large, that we should chunk (should be fine for testing)
    df = pa.table(
        {"a": np.ones(2**26, dtype="uint8")},
        schema=pa.schema([("a", pa.uint8(), False)]),
    )
    pq.write_table(df, str(tmp_path) + "/test.parquet")
    del df

    null_value = legate.core.Scalar(0, legate.core.uint8)
    arr = parquet_read_array(
        str(tmp_path) + glob_string, columns=["a"], null_value=null_value
    )
    assert cn.asarray(arr).sum() == 2**26


@pytest.mark.parametrize("columns", [None, ["b"], ["a", "b"], ["b", "a"], []])
@pytest.mark.parametrize("df", std_dataframe_set_cpu())
def test_read_polars(tmp_path, df, columns):
    if len(df.column_names) == 0:
        return

    pl = pytest.importorskip("polars")
    pq.write_table(df, tmp_path / "test.parquet")

    q = pl.scan_parquet(tmp_path / "test.parquet")
    if columns is not None:
        q = q.select(columns)

    assert_matches_polars(q, allow_exceptions=pl.exceptions.ColumnNotFoundError)


def test_multiple_files_polars(tmp_path, glob_string="/*"):
    # Simple additional test that loads multiple files
    pl = pytest.importorskip("polars")

    df = pa.table({"a": np.arange(983, dtype="int64")})
    npartitions = 100
    write_partitioned_parquet(df, tmp_path, npartitions=npartitions)
    assert len(glob.glob(str(tmp_path) + glob_string)) == npartitions

    q = pl.scan_parquet(str(tmp_path) + glob_string)
    assert_matches_polars(q)
