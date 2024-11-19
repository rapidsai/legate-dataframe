# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import cudf
import cupynumeric
from legate.core import get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.parquet import parquet_read, parquet_write


def main(tmpdir):
    """
    Example of how to use legate-dataframe, which is following the API of libcudf:
    <https://docs.rapids.ai/api/libcudf/stable/>.
    """

    # Let's start by creating a logical table from a cuDF dataframe
    # This takes a local dataframe and distribute it between Legate nodes
    df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [-1, -2, -3, -4]})
    tbl1 = LogicalTable.from_cudf(df)

    # We can write the logical table to disk using the Parquet file format.
    # The table is written into multiple files, one file per partition:
    #      /tmpdir/
    #          ├── part-0.parquet
    #          ├── part-1.parquet
    #          ├── part-2.parquet
    #          └── ...
    parquet_write(tbl1, path=tmpdir)

    # NB: since Legate execute tasks lazily, we issue a blocking fence
    #     in order to wait until all files has been written to disk.
    get_legate_runtime().issue_execution_fence(block=True)

    # Then we can read the parquet files back into a logical table. We
    # provide a Glob string that reference all the parquet files that
    # should go into the logical table.
    tbl2 = parquet_read(glob_string=f"{tmpdir}/*.parquet")

    # LogicalColumn implements the `__legate_data_interface__` interface,
    # which makes it possible for other Legate libraries, such as cuPyNumeric,
    # to operate on columns seamlessly.
    ary = cupynumeric.add(tbl1["a"], tbl2["b"])
    assert ary.sum() == 0
    ary[:] = [4, 3, 2, 1]

    # We can create a new logical column from any 1-D array like object that
    # exposes the `__legate_data_interface__` interface.
    col = LogicalColumn(ary)

    # We can create a new logical table from existing logical columns.
    LogicalTable(columns=(col, tbl2["b"]), column_names=["a", "b"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(tmpdir)
        # Since Legate execute tasks lazily, we issue a blocking fence here
        # to make sure all task has finished before `tmpdir` is removed.
        get_legate_runtime().issue_execution_fence(block=True)
