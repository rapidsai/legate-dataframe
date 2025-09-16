# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.scalar cimport Scalar as PylibcudfScalar
from pylibcudf.types cimport data_type

from .column cimport cpp_LogicalColumn


# We declare the cuda methods of cpp_LogicalColumn here, behind functions, so
# that we can compile cpp_LogicalColumn without cuda code for the CPU only version
cdef extern from "<legate_dataframe/core/column.hpp>" nogil:
    """
    using namespace legate::dataframe;
    inline LogicalColumn column_from_cudf(cudf::column_view cudf_col){
        return LogicalColumn(cudf_col);
    }
    inline LogicalColumn column_from_cudf_scalar(const cudf::scalar &cudf_scalar){
        return LogicalColumn(cudf_scalar);
    }
    inline cudf::data_type cudf_type(const LogicalColumn& col){
        return col.cudf_type();
    }
    inline std::unique_ptr<cudf::column> get_cudf(const LogicalColumn& col){
        return col.get_cudf();
    }
    inline std::unique_ptr<cudf::scalar> get_cudf_scalar(const LogicalColumn& col){
        return col.get_cudf_scalar();
    }
    """
    cdef cpp_LogicalColumn column_from_cudf "column_from_cudf"(
        column_view cudf_col
    ) except +
    cdef cpp_LogicalColumn column_from_cudf_scalar "column_from_cudf_scalar"(
        const scalar &cudf_scalar
    ) except +
    data_type cudf_type "cudf_type"(const cpp_LogicalColumn& col) except +
    unique_ptr[column] get_cudf "get_cudf"(const cpp_LogicalColumn& col) except +
    unique_ptr[scalar] get_cudf_scalar "get_cudf_scalar"(
        const cpp_LogicalColumn& col
    ) except +
