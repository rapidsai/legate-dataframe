# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


import cudf

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf.column cimport Column as PylibcudfColumn
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.scalar cimport Scalar as PylibcudfScalar
from pylibcudf.types cimport DataType

from .column cimport *


def from_cudf(col_or_scalar):
    cdef PylibcudfColumn col
    cdef PylibcudfScalar sca
    if isinstance(col_or_scalar, cudf.Series):
        col_or_scalar = col_or_scalar._column.to_pylibcudf("read")
    elif isinstance(col_or_scalar, cudf.core.column.column.ColumnBase):
        col_or_scalar = col_or_scalar.to_pylibcudf("read")
    elif isinstance(col_or_scalar, cudf.Scalar):
        col_or_scalar = col_or_scalar.device_value

    if isinstance(col_or_scalar, PylibcudfColumn):
        col = <PylibcudfColumn>col_or_scalar
        return LogicalColumn.from_handle(cpp_LogicalColumn(col.view()))
    elif isinstance(col_or_scalar, PylibcudfScalar):
        sca = <PylibcudfScalar>col_or_scalar
        return LogicalColumn.from_handle(
            cpp_LogicalColumn(dereference(sca.get()))
        )
    else:
        raise TypeError(
            "from_cudf() only supports cudf columns and device scalars."
        )


def to_cudf(LogicalColumn col):
    cdef unique_ptr[column] cudf_col = col._handle.get_cudf()
    pylibcudf_col = PylibcudfColumn.from_libcudf(move(cudf_col))
    return cudf.core.column.column.ColumnBase.from_pylibcudf(pylibcudf_col)


def to_cudf_scalar(LogicalColumn col):
    cdef unique_ptr[scalar] scal = col._handle.get_cudf_scalar()
    pylibcudf_scalar = PylibcudfScalar.from_libcudf(move(scal))
    return cudf.Scalar.from_pylibcudf(pylibcudf_scalar)


def cudf_dtype(LogicalColumn col):
    return DataType.from_libcudf(col._handle.cudf_type())
