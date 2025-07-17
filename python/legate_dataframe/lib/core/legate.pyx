# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


cdef cpp_Slice from_python_slice(slice sl):
    # Definition of this is taken directly from legate.
    if sl.step is not None and sl.step != 1:
        raise NotImplementedError(f"Unsupported slice: {sl}")

    cdef std_optional[int64_t] start = (
        OPEN
        if sl.start is None
        else std_optional[int64_t](<int64_t> sl.start)
    )
    cdef std_optional[int64_t] stop = (
        OPEN
        if sl.stop is None
        else std_optional[int64_t](<int64_t> sl.stop)
    )
    return cpp_Slice(start, stop)
