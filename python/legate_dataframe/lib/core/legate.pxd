# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport int64_t
from libcpp.optional cimport optional as std_optional


cdef extern from "legate.h":
    cpdef enum class cpp_StoreTarget "legate::mapping::StoreTarget":
        # Legate doesn't ship its Cython definitions, so define it here
        # We use the fact that Cython just allows conversion to it.
        SYSMEM
        FBMEM
        ZCMEM
        SOCKETMEM

    cdef cppclass cpp_Scalar "legate::Scalar":
        pass

    cdef cppclass cpp_Type "legate::Type":
        pass


cdef extern from "legate/data/slice.h" namespace "legate" nogil:
    cdef std_optional[int64_t] OPEN "legate::Slice::OPEN"

    cdef cppclass cpp_Slice "legate::Slice":
        cpp_Slice() except+
        cpp_Slice(std_optional[int64_t], std_optional[int64_t]) except+


cdef cpp_Slice from_python_slice(slice sl)
