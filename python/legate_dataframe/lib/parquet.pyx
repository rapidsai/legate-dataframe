# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator import dereference

from libc.stdint cimport uintptr_t
from libcpp cimport bool as cpp_bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.legate cimport cpp_Scalar, cpp_Type
from legate_dataframe.lib.core.logical_array cimport cpp_LogicalArray
from legate_dataframe.lib.core.table cimport LogicalTable, cpp_LogicalTable

import glob
import pathlib

from legate.core import LogicalArray, Scalar, Type

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/core/library.hpp>" nogil:
    cpp_bool get_prefer_eager_allocations \
        "legate::dataframe::get_prefer_eager_allocations"()

cdef extern from "<legate_dataframe/parquet.hpp>" nogil:
    void cpp_parquet_write "legate::dataframe::parquet_write"(
        cpp_LogicalTable& tbl, const string& dirpath
    ) except +
    cpp_LogicalTable cpp_parquet_read "legate::dataframe::parquet_read"(
        const vector[string]& files,
        optional[vector[string]]& columns,
        cpp_bool ignore_row_groups,
    ) except +
    cpp_LogicalArray cpp_parquet_read_array "legate::dataframe::parquet_read_array"(
        const vector[string]& files,
        optional[vector[string]]& columns,
        cpp_Scalar &null_value,
        optional[cpp_Type] &type,
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
def parquet_read(files, *, columns=None, ignore_row_groups=None) -> LogicalTable:
    """Read Parquet files into a logical table

    Parameters
    ----------
    files : str, Path, or iterable of paths
        If a string, ``glob.glob`` is used to conveniently load multiple files,
        otherwise must be a path or an iterable of paths (or strings).
    columns
        List of strings selecting a subset of columns to read.
    ignore_row_groups
        If set to ``True`` the read operation will not be chunked into row groups.
        When row groups are large, this may lead to better resource use and more
        efficient reads. Note that temporary resource use may be higher due to the
        different approaches to reading the data.

        .. note::
            The Python side default is currently set to ``True`` when
            ``LDF_PREFER_EAGER_ALLOCATIONS`` is used as it helps with streaming.
            We expect future improvements when ``ignore_row_groups``, as of now
            at least on the CPU it may not be beneficial even with large row groups.

    Returns
    -------
        The read logical table.

    See Also
    --------
    parquet_write: Write parquet data
    """
    cdef vector[string] cpp_files
    cdef vector[string] cpp_columns
    cdef optional[vector[string]] cpp_columns_opt
    cdef cpp_bool cpp_ignore_row_groups
    if ignore_row_groups is None:
        cpp_ignore_row_groups = get_prefer_eager_allocations()
    else:
        cpp_ignore_row_groups = ignore_row_groups

    if isinstance(files, str):
        files = sorted(glob.glob(files))
    elif isinstance(files, pathlib.Path):
        files = [files]

    for file in files:
        cpp_files.push_back(str(file).encode("UTF-8"))

    if columns is not None:
        for name in columns:
            cpp_columns.push_back(name.encode("UTF-8"))

        cpp_columns_opt = cpp_columns

    return LogicalTable.from_handle(
        cpp_parquet_read(cpp_files, cpp_columns_opt, cpp_ignore_row_groups)
    )


@_track_provenance
def parquet_read_array(
    files, *, columns=None, null_value=None, type=None
) -> LogicalArray:
    """Read Parquet files into a logical array

    To successfully read the files, all selected columns must have the same type
    that is compatible with legate (currently only numeric types).

    Parameters
    ----------
    files : str, Path, or iterable of paths
        If a string, ``glob.glob`` is used to conveniently load multiple files,
        otherwise must be a path or an iterable of paths (or strings).
    columns
        List of strings selecting a subset of columns to read.
    null_value : legate.core.Scalar or None
        If given (not ``None``) the result will not have a null mask
        and null values are instead replaced with this value.
    type : legate.core.Type or None
        The desired result legate type.  If given, columns are cast
        to this type.  If not given the ``dtype`` is inferred, but
        all columns must have the same one.

    Returns
    -------
        The read logical array.

    See Also
    --------
    parquet_read: Write parquet data into a table
    """
    cdef cpp_LogicalArray res_arr
    cdef vector[string] cpp_files
    cdef vector[string] cpp_columns
    cdef optional[vector[string]] cpp_columns_opt
    cdef cpp_Scalar cpp_null_value
    cdef optional[cpp_Type] cpp_type

    if isinstance(files, str):
        files = sorted(glob.glob(files))
    elif isinstance(files, pathlib.Path):
        files = [files]

    for file in files:
        cpp_files.push_back(str(file).encode("UTF-8"))

    if null_value is not None:
        if not isinstance(null_value, Scalar):
            raise TypeError("null_value must be a legate Scalar.")
        cpp_null_value = dereference(<cpp_Scalar *><uintptr_t>null_value.raw_handle)
    if type is not None:
        if not isinstance(type, Type):
            raise TypeError("type must be a legate Type.")
        cpp_type = dereference(<cpp_Type *><uintptr_t>type.raw_ptr)

    if columns is not None:
        for name in columns:
            cpp_columns.push_back(name.encode("UTF-8"))

        cpp_columns_opt = cpp_columns

    res_arr = cpp_parquet_read_array(
        cpp_files, cpp_columns_opt, cpp_null_value, cpp_type
    )
    return LogicalArray.from_raw_handle(<uintptr_t>&res_arr)
