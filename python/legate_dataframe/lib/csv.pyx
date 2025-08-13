# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp cimport bool as cpp_bool
from libcpp.memory cimport shared_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyarrow.lib cimport CDataType

from legate_dataframe.lib.core.data_type cimport as_arrow_data_type
from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

import glob
import pathlib

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/csv.hpp>" nogil:
    void cpp_csv_write "legate::dataframe::csv_write"(
        cpp_LogicalTable& tbl, const string& dirpath, char delimiter
    ) except +
    cpp_LogicalTable cpp_csv_read "legate::dataframe::csv_read"(
        const vector[string]& files,
        const vector[shared_ptr[CDataType]]& out,
        cpp_bool na_filter,
        char delimiter,
        optional[vector[string]]& names,
        optional[vector[int]]& usecols
    ) except +


@_track_provenance
def csv_write(LogicalTable tbl, path, delimiter=","):
    """Write logical table to csv files

    Each partition will be written to a separate file.

    Parameters
    ----------
    tbl : LogicalTable
        The table to write.
    path : str or pathlib.Path
        Destination directory for data.
    delimiter : str
        The field delimiter.

    Files will be created in the specified output directory using the
    convention ``part.0.csv``, ``part.1.csv``, ``part.2.csv``, ... and
    so on for each partition in the table::

        /path/to/output/
            ├── part.0.csv
            ├── part.1.csv
            ├── part.2.csv
            └── ...

    See Also
    --------
    csv_read: Read csv data
    lib.parquet.parquet_read: Read parquet data
    """
    cpp_csv_write(
        tbl._handle, str(path).encode('UTF-8'), ord(str(delimiter).encode('UTF-8'))
    )


@_track_provenance
def csv_read(
    files, *, dtypes, na_filter=True, delimiter=",", usecols=None, names=None
):
    """Read csv files into a logical table

    Parameters
    ----------
    files : str, Path, or iterable of paths
        If a string, ``glob.glob`` is used to conveniently load multiple files,
        otherwise must be a path or an iterable of paths (or strings).
    dtypes : iterable of arrow dtype-likes
        The arrow dypes to extract for each column (or a single one for all).
    na_filter: bool, optional
        Whether to detect missing values, set to ``False`` to improve performance.
    delimiter : str, optional
        The field delimiter.
    usecols : iterable of str or int or None, optional
        If given, must match `dtypes` in length and denotes column names to
        be extracted from the file.
        If passes as integers, implies the file has no header and names must
        be passed.
    names : iterable of str
        The names of the read columns, must be used with integral usecols.

    Returns
    -------
        The read logical table

    See Also
    --------
    csv_write: Write csv data
    lib.parquet.parquet_write: Write parquet data
    """
    cdef vector[string] cpp_files
    cdef vector[shared_ptr[CDataType]] cpp_dtypes
    cdef vector[string] cpp_names
    cdef vector[int] cpp_usecols
    cdef optional[vector[string]] cpp_names_opt
    cdef optional[vector[int]] cpp_usecols_opt

    if isinstance(files, str):
        files = sorted(glob.glob(files))
    elif isinstance(files, pathlib.Path):
        files = [files]

    for file in files:
        cpp_files.push_back(str(file).encode("UTF-8"))

    for dtype in dtypes:
        cpp_dtypes.push_back(as_arrow_data_type(dtype))

    # Slightly awkward.  C++ uses `names` as string `usecols`.
    if names is None:
        names = usecols  # usecols must be column names then.
        usecols = None

    if names is not None:
        for name in names:
            cpp_names.push_back(name.encode("UTF-8"))

        cpp_names_opt = cpp_names

    if usecols is not None:
        for indx in usecols:
            cpp_usecols.push_back(indx)

        cpp_usecols_opt = cpp_usecols

    return LogicalTable.from_handle(
        cpp_csv_read(
            cpp_files,
            cpp_dtypes, <cpp_bool>na_filter,
            ord(str(delimiter).encode('UTF-8')),
            cpp_names_opt,
            cpp_usecols_opt,
        )
    )
