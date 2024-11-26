# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import cudf
import cudf.core.column
import cudf.testing
import cupy
import numpy as np
import pytest

from legate_dataframe import LogicalColumn, LogicalTable


def as_cudf_dataframe(obj: Any, default_column_name: str = "data") -> cudf.DataFrame:
    """Convert an object to a cudf dataframe

    Parameters
    ----------
    obj
        Any object that can be converted to a `cudf.DataFrame` either
        through `cudf.DataFrame()`, `obj.to_cudf()`, or
        exposes a 1D array through the
        `__legate_data_interface__` interface.
    default_column_name
        The column name to use if no name are defined. This is useful when
        comparing `cudf.DataFrame` to legate objects that doesn't have column
        names such as `legate_dataframe.Column` or `cupynumeric.NDArray`.

    Returns
    -------
        The cudf dataframe
    """
    if isinstance(obj, LogicalColumn):
        obj = LogicalTable([obj], column_names=[default_column_name])
    if isinstance(obj, LogicalTable):
        return obj.to_cudf()
    if hasattr(obj, "__legate_data_interface__"):
        return LogicalTable(
            columns=[LogicalColumn(obj)], column_names=[default_column_name]
        ).to_cudf()
    if isinstance(obj, (cudf.Series, cudf.core.column.ColumnBase)):
        return cudf.DataFrame({default_column_name: obj})
    return cudf.DataFrame(obj)


def assert_frame_equal(
    left: Any,
    right: Any,
    check_index: bool = False,
    ignore_row_order: bool = False,
    default_column_name: str = "data",
    **kwargs,
) -> None:
    """Check that left and right DataFrame are equal

    Parameters
    ----------
    left
        Left dataframe to compare. Any object that can be converted to a
        cudf.DataFrame by `as_cudf_dataframe()`.
    right
        Right dataframe to compare. Any object that can be converted to a
        cudf.DataFrame by `as_cudf_dataframe()`.
    ignore_row_order
        Whether to ignore the row order or not.
    check_index
        Whether to index of each dataframe must match.
    default_column_name
        The column name to use if no name are defined. This is useful when
        comparing `cudf.DataFrame` to legate objects that doesn't have column
        names such as `legate_dataframe.Column` or `cupynumeric.NDArray`.
    kwargs
        Extra keyword arguments that are passthrough as-is to
        `cudf.testing.assert_frame_equal`

    Returns
    -------
        The extracted Legate store.
    """

    lhs = as_cudf_dataframe(left, default_column_name=default_column_name)
    rhs = as_cudf_dataframe(right, default_column_name=default_column_name)
    if ignore_row_order:
        lhs = lhs.sort_values(lhs.columns, ignore_index=not check_index)
        rhs = rhs.sort_values(rhs.columns, ignore_index=not check_index)
    if not check_index:
        lhs = lhs.reset_index(drop=True)
        rhs = rhs.reset_index(drop=True)

    cudf.testing.assert_frame_equal(
        left=lhs,
        right=rhs,
        **kwargs,
    )


def get_empty_series(dtype, nullable: bool) -> cudf.Series:
    """Create an empty cudf series

    Parameters
    ----------
    dtype
        The dtype of the new series.
    nullable
        Whether to add a null mask to the empty series.

    Returns
    -------
        The new empty series
    """
    ret = cudf.Series([], dtype=dtype)
    if nullable:
        ret._column.set_mask(np.empty(shape=(0,), dtype="uint8"))
    return ret


def std_dataframe_set() -> List[cudf.DataFrame]:
    """Return the standard test set of dataframes

    Used throughout the test suite to check against supported data types

    Returns
    -------
        List of dataframes
    """
    return [
        cudf.DataFrame({"a": cupy.arange(10000, dtype="int64")}),
        cudf.DataFrame(
            {
                "a": cupy.arange(10000, dtype="int32"),
                "b": cupy.arange(-10000, 0, dtype="float64"),
            }
        ),
        cudf.DataFrame({"a": ["a", "bb", "ccc"]}),
        cudf.DataFrame(
            {
                "a": get_empty_series(dtype=int, nullable=True),
                "b": get_empty_series(dtype=float, nullable=True),
            }
        ),
    ]


def get_column_set(dtypes, nulls=True):
    """Return a set of columns with the given dtypes

    Can be used to test a pytest fixture to generate a set of columns.

    Parameters
    ----------
    dtypes : sequence of dtypes
        The dtypes for the returned columns cudf must support casting
        integers to it.
    nulls : boolean, optional
        If set  to``False`` the returned columns do not contain booleans.

    Yields
    ------
    parameter : pytest.param
        Pytest parameters each containing a columns.
    """
    data = np.arange(-1000, 1000)
    np.random.seed(0)

    for dtype in dtypes:
        series = cudf.Series(data).astype(dtype)
        if nulls:
            series = series.mask(np.random.randint(2, size=len(series), dtype=bool))

        yield pytest.param(series._column, id=f"col({dtype}, nulls={nulls})")
