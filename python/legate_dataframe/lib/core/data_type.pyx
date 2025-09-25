# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp.memory cimport shared_ptr

from pyarrow.lib cimport CDataType, pyarrow_unwrap_data_type

import legate.core
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
    if type(data_type_like).__module__ == "pylibcudf.types":
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


cdef dict map_to_legate = {
    pa.int8(): legate.core.int8,
    pa.int16(): legate.core.int16,
    pa.int32(): legate.core.int32,
    pa.int64(): legate.core.int64,
    pa.uint8(): legate.core.uint8,
    pa.uint16(): legate.core.uint16,
    pa.uint32(): legate.core.uint32,
    pa.uint64(): legate.core.uint64,
    pa.float32(): legate.core.float32,
    pa.float64(): legate.core.float64,
    pa.bool_(): legate.core.bool_,
    pa.string(): legate.core.string_type,
    pa.large_string(): legate.core.string_type,
}


cdef bint is_legate_compatible(arrow_type: pa.DataType):
    """Check if a datatype is a native legate datatype. For now, we do
    this by simply hardcoding the numeric ones plus bool and string.
    """
    return arrow_type in map_to_legate
