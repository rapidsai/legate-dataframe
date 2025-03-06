# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from cudf._lib.types cimport underlying_type_t_type_id
from pylibcudf.libcudf.types cimport type_id
from pylibcudf.types cimport DataType
from pylibcudf.types cimport data_type as cpp_cudf_type

import cudf
from cudf._lib.types import PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES, dtype_to_pylibcudf_type
from numpy.typing import DTypeLike


cdef cpp_cudf_type as_data_type(data_type_like: DTypeLike):
    """Get data type from object

    Parameters
    ----------
    data_type_like
        A Python object that is convertible to a cudf datatype.

    Returns
    -------
        Coerced C++ cudf type.
    """

    cdef DataType dtype = dtype_to_pylibcudf_type(cudf.dtype(data_type_like))
    return dtype.c_obj


cdef cpp_cudf_type_to_cudf_dtype(cpp_cudf_type libcudf_type):
    """Convert a libcudf data type to a Python cudf/numpy data type

    Parameters
    ----------
    libcudf_type
        A libcudf data type.

    Returns
    -------
    DtypeObj
        Coerced Python data type.
    """

    cdef type_id tid = libcudf_type.id()

    if tid == type_id.LIST:
        raise NotImplementedError("LIST data type not implemented")
    elif tid == type_id.STRUCT:
        raise NotImplementedError("LIST data type not implemented")
    elif tid == type_id.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION,
            scale=-libcudf_type.scale()
        )
    elif tid == type_id.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION,
            scale=-libcudf_type.scale()
        )
    elif tid == type_id.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION,
            scale=-libcudf_type.scale()
        )
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
            <underlying_type_t_type_id>(tid)
        ]


cdef bint is_legate_compatible(cpp_cudf_type libcudf_type):
    """Check if a datatype is a native legate datatype. For now, we do
    this by simply hardcoding the numeric ones plus bool and string.
    """
    cdef type_id tid = libcudf_type.id()

    if tid in (
        type_id.INT8, type_id.INT16, type_id.INT32, type_id.INT64,  # signed
        type_id.UINT8, type_id.UINT16, type_id.UINT32, type_id.UINT64,  # unsigned
        type_id.FLOAT32, type_id.FLOAT64,  # floats
        type_id.BOOL8,  type_id.STRING
    ):
        return True
    else:
        return False
