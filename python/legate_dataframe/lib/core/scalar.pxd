# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.scalar.scalar cimport scalar as cudf_cpp_scalar


cdef extern from "legate.h" nogil:
    cdef cppclass cpp_legate_Scalar "legate::Scalar":
        cpp_legate_Scalar()

cdef extern from "<legate_dataframe/core/scalar.hpp>" nogil:
    cdef cppclass cpp_ScalarArg "legate::dataframe::ScalarArg":
        cpp_ScalarArg() except +
        cpp_ScalarArg(cpp_legate_Scalar legate_scalar) except +
        cpp_ScalarArg(cudf_cpp_scalar cudf_scalar) except +

        unique_ptr[cudf_cpp_scalar] get_cudf() except +
        bool is_null()

cdef cpp_ScalarArg cpp_scalar_arg_from_python(pyscalar)
