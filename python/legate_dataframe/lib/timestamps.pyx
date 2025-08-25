# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3


from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from pyarrow.lib cimport CDataType

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_arrow_data_type

from numpy.typing import DTypeLike

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/timestamps.hpp>" namespace "legate::dataframe":
    cpp_LogicalColumn cpp_to_timestamps "to_timestamps"(
        const cpp_LogicalColumn& input,
        shared_ptr[CDataType] timestamp_type,
        string format,
    ) except +

    cpp_LogicalColumn cpp_extract_timestamp_component "extract_timestamp_component"(
        const cpp_LogicalColumn& input,
        string component,
    ) except +


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
            as_arrow_data_type(timestamp_type),
            str(format_pattern).encode('UTF-8')
        )
    )


@_track_provenance
def extract_timestamp_component(
    LogicalColumn col,
    component: str,
) -> LogicalColumn:
    """
    Extract part of the timestamp as int16.

    Parameters
    ----------
    col : LogicalColumn
        Column of timestamps
    component
        The component which to extract. A string like "year", "month",
        "day", "millisecond" etc. See arrow documentation for
        "Temporal component extraction" for a full list.

    Returns
    -------
        New int64 column

    Notes
    -----
    Unlike `pandas` and `cudf`, this function counts the days of the
    week as Monday-Sunday being 1-7 and ``microsecond_fraction`` does not
    include milliseconds.

    """
    return LogicalColumn.from_handle(
        cpp_extract_timestamp_component(col._handle, component.encode('UTF-8'))
    )
