# Copyright (c) 2024-2025, NVIDIA CORPORATION
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
import math

import cudf
import numpy as np
import pyarrow as pa
import pytest
from legate.core import TaskTarget, get_legate_runtime, get_machine
from pyarrow import csv

from legate_dataframe import LogicalTable
from legate_dataframe.lib.csv import csv_read, csv_write
from legate_dataframe.testing import (
    assert_arrow_table_equal,
    assert_frame_equal,
    assert_matches_polars,
    std_dataframe_set_cpu,
)


def write_partitioned_csv(table, path, npartitions=1):
    if table.num_rows == 0:
        csv.write_csv(table, f"{path}/part-0.csv")
    partition_size = int(math.floor(table.num_rows / npartitions))
    for i in range(npartitions):
        start = i * partition_size
        end = min((i + 1) * partition_size, table.num_rows)
        if i == npartitions - 1:
            # Last partition may be larger if not evenly divisible
            end = table.num_rows
        if start >= end:
            break
        partition = table[start:end]
        csv.write_csv(partition, f"{path}/part-{i}.csv")


@pytest.mark.parametrize("df", std_dataframe_set_cpu())
def test_write(tmp_path, df):
    tbl = LogicalTable.from_arrow(df)

    csv_write(tbl, path=tmp_path)
    get_legate_runtime().issue_execution_fence(block=True)

    # Read the files back with pyarrow then compare with the original
    files = sorted(glob.glob(str(tmp_path) + "/*.csv"))
    tables = [
        csv.read_csv(file, convert_options=csv.ConvertOptions(column_types=df.schema))
        for file in files
    ]

    # Concatenate the tables
    combined_table = pa.concat_tables(tables)

    assert_arrow_table_equal(combined_table, df)


@pytest.mark.parametrize("df", std_dataframe_set_cpu())
def test_read(tmp_path, df, npartitions=2):
    filenames = str(tmp_path) + "/*.csv"
    write_partitioned_csv(df, tmp_path, npartitions=npartitions)
    tbl = csv_read(filenames, dtypes=df.schema.types)
    assert_arrow_table_equal(tbl.to_arrow(), df)


def test_read_single_rows(tmp_path):
    filenames = str(tmp_path) + "/*.csv"
    df = pa.table({"a": np.arange(1, dtype="int64")})
    write_partitioned_csv(df, tmp_path, npartitions=1)
    tbl = csv_read(filenames, dtypes=df.schema.types)
    assert_arrow_table_equal(tbl.to_arrow(), df)


def test_read_single_many_columns(tmp_path):
    if get_machine().count(TaskTarget.GPU) > 2:
        pytest.skip(reason="As of 25.03 test seems to need more ZCMEM with more GPUs.")
    # Legate used to have a limit on number of returns which limnits the
    # number of columns (<25.03).  So make sure we support ~1000.
    file = tmp_path / "file.csv"
    # Write a file with many columns (and a few rows)
    ncols = 1000
    for i in range(5):
        file.write_text(",".join([str(i) for i in range(ncols)]) + "\n")

    # Guard by execution fence for sports (the skip above seems needed...)
    get_legate_runtime().issue_execution_fence(block=True)
    csv_read(file, dtypes=["int64"] * ncols)
    get_legate_runtime().issue_execution_fence(block=True)


def test_read_many_files_per_rank(tmp_path):
    # Use uneven number to test splitting
    filenames = str(tmp_path) + "/*.csv"
    df = pa.table({"a": np.arange(983, dtype="int64")})
    npartitions = 100
    write_partitioned_csv(df, tmp_path, npartitions=npartitions)
    # Test that we really have many files hoped for:
    assert len(glob.glob(filenames)) == npartitions
    tbl = csv_read(glob.glob(filenames), dtypes=df.schema.types)

    assert_arrow_table_equal(tbl.to_arrow().sort_by("a"), df)


@pytest.mark.parametrize("delimiter", [",", "|"])
def test_readwrite_dates(tmp_path, delimiter):
    df = cudf.DataFrame(
        {"a": ["2010-06-19T13:15", "2011-06-19T13:25", "2010-07-19T13:35"]}
    ).astype("datetime64[ns]")

    tbl = LogicalTable.from_cudf(df)
    csv_write(tbl, tmp_path, delimiter=delimiter)
    get_legate_runtime().issue_execution_fence(block=True)

    read_tbl = csv_read(
        str(tmp_path) + "/*", dtypes=["datetime64[ns]"], delimiter=delimiter
    )

    assert_frame_equal(read_tbl, df)


def test_trailing_nulls(tmp_path):
    # If we read a large file where a chunk (or all) of rows end in nulls
    # then csv may not be clear about the number of columns contained, so
    # test that we ensure the correct number:
    df = cudf.DataFrame(
        {"a": list(range(100)), "b": cudf.Series([None] * 100, dtype="int64")}
    )

    df.to_csv(tmp_path / "tmp.csv", index=False)

    read_tbl = csv_read(str(tmp_path) + "/*", dtypes=["int64", "int64"])
    assert_frame_equal(read_tbl, df)


def test_wrong_number_of_dtypes(tmp_path):
    df = cudf.DataFrame({"a": [1, 2, 3, 4]})
    df.to_csv(tmp_path / "tmp.csv", index=False)
    with pytest.raises(ValueError, match="number of columns in csv"):
        csv_read(str(tmp_path) + "/*", dtypes=["int64", "int64"])


def test_usecols(tmp_path):
    df = cudf.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 3, 4]})
    df.to_csv(tmp_path / "tmp.csv", index=False)

    with pytest.raises(ValueError, match="usecols, names, and dtypes"):
        csv_read(str(tmp_path) + "/*", dtypes=["int64", "int64"], usecols=["a"])

    with pytest.raises(ValueError, match="column 'bad-name' not found"):
        csv_read(str(tmp_path) + "/*", dtypes=["int64"], usecols=["bad-name"])

    read_tbl = csv_read(
        str(tmp_path) + "/*", dtypes=["int64", "float64"], usecols=["b", "c"]
    )
    assert_frame_equal(read_tbl, df[["b", "c"]].astype({"c": "float64"}))

    read_tbl = csv_read(
        str(tmp_path) + "/*", dtypes=["int64", "float64"], usecols=["c", "b"]
    )
    assert_frame_equal(read_tbl, df[["b", "c"]].astype({"b": "float64"}))


def test_usecols_and_names_no_header(tmp_path):
    df = cudf.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 3, 4]})
    file_path = tmp_path / "tmp.csv"
    df.to_csv(file_path, index=False)

    # Remove the header (first line):
    file_path.write_text("\n".join(file_path.read_text().split("\n")[1:]))

    with pytest.raises(ValueError, match="usecols, names, and dtypes"):
        csv_read(
            str(tmp_path) + "/*", dtypes=["int64", "int64"], usecols=[0, 2], names=["a"]
        )

    read_tbl = csv_read(
        str(tmp_path) + "/*",
        dtypes=["int64", "float64"],
        usecols=[1, 2],
        names=["b", "c"],
    )
    assert_frame_equal(read_tbl, df[["b", "c"]].astype({"c": "float64"}))

    read_tbl = csv_read(
        str(tmp_path) + "/*",
        dtypes=["int64", "float64"],
        usecols=[2, 1],
        names=["c", "b"],
    )
    assert_frame_equal(read_tbl, df[["b", "c"]].astype({"b": "float64"}))


@pytest.mark.skipif(
    get_machine().count(TaskTarget.GPU) == 0,
    reason="Arrow does not support this na_filter option",
)
def test_na_filter_false(tmp_path):
    df = cudf.DataFrame({"a": [1, 2, 3, 4]})
    df.to_csv(tmp_path / "tmp.csv", index=False)
    read_tbl = csv_read(str(tmp_path) + "/*", dtypes=["int64"], na_filter=False)
    # NA filter does (currently) not mean that the column isn't set nullable:
    # assert not read_tbl["a"].to_cudf().nullable
    assert_frame_equal(read_tbl, df)

    # And it fails if we try do not allow nulls.
    df = cudf.DataFrame(
        {
            "a": cudf.Series([None, 1, 2, None, None], dtype="int64"),
            "b": cudf.Series([None, 1, 3, 4, None], dtype="int64"),
        }
    )
    df.to_csv(tmp_path / "tmp.csv", index=False)

    # Cudf reads empty integer columns as 0 with na_filter=False
    read_tbl = csv_read(
        str(tmp_path) + "/*", dtypes=["int64", "int64"], na_filter=False
    ).to_cudf()
    assert_frame_equal(read_tbl, df.fillna(0))


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_num_rows_split(tmp_path, offset):
    # Sanity check that file splitting boundaries work correctly.
    # We use a different number of bytes for a so that the two runs must split
    # at one byte different boundaries w.r.t. the rows containing "1\n".
    with open(tmp_path / "tmp.csv", "w") as f:
        f.write("a" * offset + "\n1" * 1000)

    read_tbl = csv_read(str(tmp_path) + "/*", dtypes=["int64"])
    assert read_tbl.num_rows() == 1000


def test_read_polars(tmp_path):
    # Test basic polars csv reading.
    pl = pytest.importorskip("polars")

    df = cudf.DataFrame({"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 3, 4]})
    df.to_csv(tmp_path / "tmp.csv", index=False)

    q = pl.scan_csv(tmp_path / "tmp.csv")
    assert_matches_polars(q)
    # Check that selection push-through works
    q = q.select([pl.col("a"), pl.col("c")])
    assert_matches_polars(q)


def test_read_polars_multiple_files(tmp_path):
    # Simple additional test that loads multiple files.
    pl = pytest.importorskip("polars")

    filenames = str(tmp_path) + "/*.csv"
    df = pa.table({"a": np.arange(983, dtype="int64")})
    npartitions = 100
    write_partitioned_csv(df, tmp_path, npartitions=npartitions)
    # Test that we really have many files hoped for:
    assert len(glob.glob(filenames)) == npartitions
    q = pl.scan_csv(filenames)
    assert_matches_polars(q)
