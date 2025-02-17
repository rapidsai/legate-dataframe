# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.legate cimport cpp_StoreTarget
from legate_dataframe.lib.core.legate_task cimport cpp_AutoTask
from legate_dataframe.lib.core.logical_array cimport cpp_LogicalArray


cdef extern from "<legate_dataframe/core/column.hpp>" nogil:
    cdef cppclass cpp_LogicalColumn "legate::dataframe::LogicalColumn":
        cpp_LogicalColumn() except +
        cpp_LogicalColumn(cpp_LogicalArray logical_array) except +
        cpp_LogicalColumn(column_view cudf_col) except +
        cpp_LogicalColumn(scalar &cudf_scalar) except +

        @staticmethod
        cpp_LogicalColumn empty_like(const cpp_LogicalColumn& other) except +

        size_t num_rows() except +
        cpp_LogicalArray get_logical_array() except +
        unique_ptr[column] get_cudf() except +
        unique_ptr[scalar] get_cudf_scalar() except +
        string repr(size_t max_num_items) except +
        bool is_scalar() noexcept
        data_type cudf_type() except +
        void offload_to(cpp_StoreTarget target_mem) except +

    void cpp_add_next_input "legate::dataframe::argument::add_next_input"(
        const cpp_AutoTask &task,
        const cpp_LogicalColumn &col
    ) except +

    void cpp_add_next_output "legate::dataframe::argument::add_next_output"(
        const cpp_AutoTask &task,
        const cpp_LogicalColumn &col
    ) except +


cdef class LogicalColumn:
    cdef cpp_LogicalColumn _handle

    @staticmethod
    cdef LogicalColumn from_handle(cpp_LogicalColumn handle)
