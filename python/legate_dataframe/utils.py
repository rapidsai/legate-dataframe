# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

from legate.core import LogicalArray
from legate.core import track_provenance as _track_provenance_legate


def get_logical_array(obj: Any) -> LogicalArray:
    """Extracts a logical array object implementing the legate data interface

    The object must expose a single logical array.

    Parameters
    ----------
    obj
        Objects that exposes `__legate_data_interface__` interface.

    Returns
    -------
        The extracted Legate store.
    """
    if isinstance(obj, LogicalArray):
        return obj

    # Extract the very first store and make sure it is the only one
    arrays = obj.__legate_data_interface__["data"].values()
    if len(arrays) != 1:
        raise ValueError("object must expose a single logical array")
    return next(iter(arrays))


def _track_provenance(func):
    """
    Private decorator to add "provenance" tracking to all Python side
    functions which end up calling legate tasks.
    All calls which directly launch tasks should be decorated.

    This e.g. adds Python line number to profiling results.  Similar to
    cupynumeric, we use `functools.update_wrapper` which the legate core
    version did not at the time of writing.
    """
    wrapped_func = _track_provenance_legate()(func)
    functools.update_wrapper(wrapped_func, func)
    return wrapped_func
