# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from pylibcudf.types cimport data_type as cpp_cudf_type


cdef cpp_cudf_type as_data_type(data_type_like)

cdef cpp_cudf_type_to_cudf_dtype(cpp_cudf_type libcudf_type)
