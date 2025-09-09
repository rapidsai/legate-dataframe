# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string

from pyarrow.lib cimport CArray, CDataType, CScalar
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.legate cimport cpp_Slice, cpp_StoreTarget
from legate_dataframe.lib.core.legate_task cimport cpp_AutoTask
from legate_dataframe.lib.core.logical_array cimport cpp_LogicalArray


cdef extern from "<legate_dataframe/core/column.hpp>" nogil:
    cdef cppclass cpp_LogicalColumn "legate::dataframe::LogicalColumn":
        cpp_LogicalColumn(column_view cudf_col) except +
        cpp_LogicalColumn(scalar &cudf_scalar) except +
        unique_ptr[column] get_cudf() except +
        unique_ptr[scalar] get_cudf_scalar() except +
        data_type cudf_type() except +


cdef class LogicalColumn:
    cdef cpp_LogicalColumn _handle

    @staticmethod
    cdef LogicalColumn from_handle(cpp_LogicalColumn handle)
