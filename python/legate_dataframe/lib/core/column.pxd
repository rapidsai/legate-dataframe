# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string

from pyarrow.lib cimport CArray, CDataType, CScalar

from legate_dataframe.lib.core.legate cimport cpp_Slice, cpp_StoreTarget
from legate_dataframe.lib.core.legate_task cimport cpp_AutoTask
from legate_dataframe.lib.core.logical_array cimport cpp_LogicalArray


cdef extern from "<legate_dataframe/core/column.hpp>" nogil:
    cdef cppclass cpp_LogicalColumn "legate::dataframe::LogicalColumn":
        cpp_LogicalColumn() except +
        cpp_LogicalColumn(
            cpp_LogicalArray logical_array,
            shared_ptr[CDataType] type,
            bool scalar,
        ) except +
        cpp_LogicalColumn(shared_ptr[CArray] arrow_array) except +
        cpp_LogicalColumn(shared_ptr[CScalar] arrow_array) except +

        @staticmethod
        cpp_LogicalColumn empty_like(const cpp_LogicalColumn& other) except +

        size_t num_rows() except +
        cpp_LogicalArray get_logical_array() except +
        shared_ptr[CArray] get_arrow() except +
        bool is_scalar() noexcept
        shared_ptr[CDataType] arrow_type() except +
        void offload_to(cpp_StoreTarget target_mem) except +
        cpp_LogicalColumn slice(cpp_Slice slice) except +
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
