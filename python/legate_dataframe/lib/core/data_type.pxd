# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport shared_ptr

from pyarrow.lib cimport CDataType
from pylibcudf.types cimport DataType
from pylibcudf.types cimport data_type as cpp_cudf_type


cdef shared_ptr[CDataType] as_arrow_data_type(data_type_like)

cdef bint is_legate_compatible(arrow_type: pa.DataType)
