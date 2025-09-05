# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp.memory cimport shared_ptr

from pyarrow.lib cimport CDataType, pyarrow_unwrap_data_type

import pyarrow as pa


cdef shared_ptr[CDataType] as_arrow_data_type(data_type_like):
    """Get data type from object

    Parameters
    ----------
    data_type_like
        A Python object that is convertible to an arrow datatype.

    Returns
    -------
        Coerced C++ arrow type.
    """
    cdef shared_ptr[CDataType] d_type

    if isinstance(data_type_like, pa.DataType):
        d_type = pyarrow_unwrap_data_type(data_type_like)
        return d_type
    if "cudf" in type(data_type_like).__name__:
        from pylibcudf.interop import to_arrow
        data_type_like = to_arrow(data_type_like)
        d_type = pyarrow_unwrap_data_type(data_type_like)
        return d_type

    # try numpy dtype
    try:
        d_type = pyarrow_unwrap_data_type(
            pa.from_numpy_dtype(data_type_like)
        )
        return d_type
    except Exception as e:
        raise TypeError(
            f"Cannot convert {data_type_like} to an arrow data type."
        ) from e


cdef bint is_legate_compatible(arrow_type: pa.DataType):
    """Check if a datatype is a native legate datatype. For now, we do
    this by simply hardcoding the numeric ones plus bool and string.
    """
    if arrow_type in (
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.float32(),
        pa.float64(),
        pa.bool_(),
        pa.string(),
    ):
        return True
    else:
        return False
