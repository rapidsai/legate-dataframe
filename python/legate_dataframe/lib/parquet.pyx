# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

import pathlib

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/parquet.hpp>" nogil:
    void cpp_parquet_write "legate::dataframe::parquet_write"(
        cpp_LogicalTable& tbl, const string& dirpath
    ) except +
    cpp_LogicalTable cpp_parquet_read "legate::dataframe::parquet_read"(
        const string& glob_string,
        optional[vector[string]]& columns,
    ) except +


@_track_provenance
def parquet_write(LogicalTable tbl, path: pathlib.Path | str) -> None:
    """Write logical table to Parquet files

    Each partition will be written to a separate file.

    Parameters
    ----------
    tbl : LogicalTable
        The table to write.
    path : str or pathlib.Path
        Destination directory for data.

    Files will be created in the specified output directory using the
    convention ``part.0.parquet``, ``part.1.parquet``, ``part.2.parquet``, ... and
    so on for each partition in the table::

        /path/to/output/
            ├── part.0.parquet
            ├── part.1.parquet
            ├── part.2.parquet
            └── ...

    See Also
    --------
    parquet_read: Read parquet data
    """
    cpp_parquet_write(tbl._handle, str(path).encode('UTF-8'))


@_track_provenance
def parquet_read(glob_string: pathlib.Path | str, *, columns=None) -> LogicalTable:
    """Read Parquet files into a logical table

    Parameters
    ----------
    glob_string : str or pathlib.Path
        The glob string to specify the Parquet files. All glob matches
        must be valid Parquet files and have the same LogicalTable data
        types. See <https://linux.die.net/man/7/glob>.
    columns
        List of strings selecting a subset of columns to read.

    Returns
    -------
        The read logical table.

    See Also
    --------
    parquet_write: Write parquet data
    """
    cdef vector[string] cpp_columns
    cdef optional[vector[string]] cpp_columns_opt

    if columns is not None:
        for name in columns:
            cpp_columns.push_back(name.encode("UTF-8"))

        cpp_columns_opt = cpp_columns

    return LogicalTable.from_handle(
        cpp_parquet_read(str(glob_string).encode('UTF-8'), cpp_columns_opt)
    )
