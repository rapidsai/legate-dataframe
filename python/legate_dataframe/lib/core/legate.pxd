# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


cdef extern from "legate.h":
    cpdef enum class cpp_StoreTarget "legate::mapping::StoreTarget":
        # Legate doesn't ship its Cython definitions, so define it here
        # We use the fact that Cython just allows conversion to it.
        SYSMEM
        FBMEM
        ZCMEM
        SOCKETMEM
