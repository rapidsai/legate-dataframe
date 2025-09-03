# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/search.hpp>" nogil:
    cpp_LogicalColumn cpp_contains "contains"(
        const cpp_LogicalColumn& haystack,
        const cpp_LogicalColumn& neeldes,
    ) except +


@_track_provenance
def contains(
    LogicalColumn haystack,
    LogicalColumn needles,
) -> LogicalColumn:
    """Check if haystack contains the values in needles.

    The result will contain boolean values indicating whether each element in
    the input column exists in the set of values.
    This is an elementwise ``needles[i] in haystack``.

    Parameters
    ----------
    haystack
        Column of values to search against. This column is currently broadcast to
        all workers and assumed to be small.
    needles
        Column of values to check if they exist in the haystack.

    Returns
    -------
        Boolean column indicating which values exist in the set, has the same
        size and nullability as haystack.

    Raises
    ------
    ValueError
        If the input columns have different types.

    """
    return LogicalColumn.from_handle(
        cpp_contains(haystack._handle, needles._handle)
    )
