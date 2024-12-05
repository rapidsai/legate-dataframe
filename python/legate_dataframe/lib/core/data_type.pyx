# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from cudf._lib.types cimport underlying_type_t_type_id
from pylibcudf.types cimport DataType
from pylibcudf.types cimport data_type as cpp_cudf_type

import cudf
import pylibcudf
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

    cdef DataType type_ = DataType.from_libcudf(libcudf_type)
    tid = type_.id()

    if tid == pylibcudf.TypeId.LIST:
        raise NotImplementedError("LIST data type not implemented")
    elif tid == pylibcudf.TypeId.STRUCT:
        raise NotImplementedError("LIST data type not implemented")
    elif tid == pylibcudf.TypeId.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == pylibcudf.TypeId.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    elif tid == pylibcudf.TypeId.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION,
            scale=-type_.scale()
        )
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[
            <underlying_type_t_type_id>(tid)
        ]
