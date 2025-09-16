# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport int32_t
from libcpp.string cimport string
from libcpp.vector cimport vector

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.legate cimport cpp_Slice, cpp_StoreTarget


cdef extern from "<legate_dataframe/core/table.hpp>" nogil:
    cdef cppclass cpp_LogicalTable "legate::dataframe::LogicalTable":
        cpp_LogicalTable() except +
        cpp_LogicalTable(
            vector[cpp_LogicalColumn] columns,
            const vector[string]& column_names
        ) except +

        int32_t num_columns() except +
        size_t num_rows() except +
        cpp_LogicalColumn& get_column(size_t column_index) except +
        cpp_LogicalColumn& get_column(const string& column_index) except +
        vector[string] get_column_name_vector() except +
        cpp_LogicalTable select(vector[string] columns) except +
        cpp_LogicalTable select(vector[size_t] columns) except +
        void offload_to(cpp_StoreTarget target_mem) except +
        cpp_LogicalTable slice(cpp_Slice slice) except +


cdef class LogicalTable:
    cdef cpp_LogicalTable _handle

    @staticmethod
    cdef LogicalTable from_handle(cpp_LogicalTable handle)

    cdef LogicalColumn get_column_by_index(self, size_t idx)
