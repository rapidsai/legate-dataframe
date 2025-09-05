# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/strings.hpp>" nogil:
    cpp_LogicalColumn cpp_match "legate::dataframe::strings::match"(
        const string& match_func,
        const cpp_LogicalColumn& values,
        const string& pattern,
    ) except +


@_track_provenance
def match(str match_func, LogicalColumn column, str pattern) -> LogicalColumn:
    """Check if strings match a given pattern.

    Parameters
    ----------
    match_func
        The type of matching to perform: "starts_with", "ends_with", "match_substring",
        or "match_substring_regex".
        (Note that the "match_substring*" check for containment not full matches.)
    column
        The column of string values to check
    pattern
        The pattern string to check for. A regular expression for
        "match_substring_regex".

    Returns
    -------
        A boolean column indicating which values match the pattern

    """
    return LogicalColumn.from_handle(
        cpp_match(match_func.encode('utf-8'), column._handle, pattern.encode('utf-8'))
    )
