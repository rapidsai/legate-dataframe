# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from legate_dataframe.lib.core.column cimport LogicalColumn


cdef LogicalColumn cpp_scalar_col_from_python(pyscalar)
