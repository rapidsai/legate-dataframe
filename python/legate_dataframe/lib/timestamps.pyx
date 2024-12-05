# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libc.stdint cimport int32_t
from libcpp.string cimport string

from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_data_type

from numpy.typing import DTypeLike

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/timestamps.hpp>" namespace "legate::dataframe":
    cpdef enum class DatetimeComponent(int32_t):
        """Datetime component to extract, can be one of:
        ``year``, ``months``, ``day``, ``weekday``, ``hour``, ``minute``,
        ``second``, ``millisecond_fraction``, ``microsecond_fraction``,
        ``nanosecond_fraction``, and ``day_of_year``.
        """
        year
        month
        day
        weekday
        hour
        minute
        second
        millisecond_fraction
        microsecond_fraction
        nanosecond_fraction
        day_of_year

    cpp_LogicalColumn cpp_to_timestamps "to_timestamps"(
        const cpp_LogicalColumn& input,
        data_type timestamp_type,
        string format,
    )

    cpp_LogicalColumn cpp_extract_timestamp_component "extract_timestamp_component"(
        const cpp_LogicalColumn& input,
        DatetimeComponent component,
    )


@_track_provenance
def to_timestamps(
    LogicalColumn col,
    timestamp_type: DTypeLike,
    format_pattern: str,
) -> LogicalColumn:
    """
    Converting a strings column into timestamps using the provided format pattern.

    The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,
    %M,%S,%f,%z".

    Please see :external:cpp:func:`to_timestamps` for details.

    .. warning::
        Invalid formats are not checked, the format pattern must be well
        defined as per the C++ API.

    Parameters
    ----------
    col
        Strings instance for this operation
    timestamp_type
        The timestamp type used for creating the output column
    format_pattern
        String specifying the timestamp format in strings

    Returns
    -------
        New datetime column

    Raises
    ------
    RuntimeError: if timestamp_type is not a timestamp type.

    """
    return LogicalColumn.from_handle(
        cpp_to_timestamps(
            col._handle,
            as_data_type(timestamp_type),
            str(format_pattern).encode('UTF-8')
        )
    )


@_track_provenance
def extract_timestamp_component(
    LogicalColumn col,
    field: DatetimeComponent,
) -> LogicalColumn:
    """
    Extract part of the timestamp as int16.

    Parameters
    ----------
    col : LogicalColumn
        Column of timestamps
    field
        The field/resolution to extract.  Must be specified as a
        ``DatetimeComponent``.

    Returns
    -------
        New int16 column

    Notes
    -----
    Unlike `pandas` and `cudf`, this function counts the days of the
    week as Monday-Sunday being 1-7 and ``microsecond_fraction`` does not
    include milliseconds.

    """
    return LogicalColumn.from_handle(
        cpp_extract_timestamp_component(col._handle, field)
    )
