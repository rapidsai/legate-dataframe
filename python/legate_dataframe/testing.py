# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Any, List

import legate.core
import numpy as np
import pyarrow as pa
import pytest
from legate.core import TaskTarget, get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable


def as_cudf_dataframe(obj: Any, default_column_name: str = "data") -> Any:
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
    import cudf
    import cudf.core.column

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


def replace_nans(table: pa.Table) -> pa.Table:
    new_table = pa.table(table)
    for i, name in enumerate(table.schema.names):
        type = table.schema.field(name).type
        if pa.types.is_floating(type):
            # Replace NaNs with 0.0 for floating point columns
            nan_mask = pa.compute.is_nan(table.column(name)).combine_chunks()
            new_table = new_table.set_column(
                i,
                name,
                pa.compute.replace_with_mask(
                    table.column(name), nan_mask, pa.scalar(0.0, type=type)
                ),
            )
    return new_table


# This function compares two Arrow tables for equality
# It compares NaN values as equal - arrows default behavior is to not compare NaNs as equal
# https://github.com/apache/arrow/issues/22446
# It ignores the nullable field in the schema as this is inconsistent in arrow
def assert_arrow_table_equal(left: pa.Table, right: pa.Table, approx=False) -> None:
    # arrow has an annoying nullable attribute in its schema that is not well respected by its various functions
    # i.e. it is possible to have a non-nullable column with null values without problems
    # Set the nullable attribute to match the left table

    assert left.schema.names == right.schema.names
    fields = []
    for name in left.schema.names:
        left_field = left.schema.field(name)
        right_field = right.schema.field(name)
        type_ = right_field.type
        # Accept if there is a mismatch with large vs. non-large strings
        if type_ == pa.large_string() and left_field.type == pa.string():
            type_ = pa.string()
        fields.append(pa.field(right_field.name, type_, left_field.nullable))
    new_schema = pa.schema(fields)
    left_copy = replace_nans(pa.table(left, schema=new_schema))
    right_copy = replace_nans(pa.table(right, schema=new_schema))

    if not approx:
        assert left_copy.equals(
            right_copy
        ), f"Arrow tables are not equal:\n{left}\n{right}"
    else:
        assert left_copy.schema == right_copy.schema
        for left_col, right_col in zip(left_copy.columns, right_copy.columns):
            assert left_col.is_valid() == right_col.is_valid()
            assert left_col.type == right_col.type  # probably already checked in schema
            if left_col.type == pa.string():
                # For string columns, we can compare the values directly
                assert left_col.equals(right_col)
            else:
                np.testing.assert_array_almost_equal(
                    left_col.drop_null().to_numpy(), right_col.drop_null().to_numpy()
                )


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
    import cudf.testing

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


def assert_matches_polars(query: Any, allow_exceptions=(), approx=False) -> None:
    """Check that a polars query is equivalent when collected via
    legate or polars.

    Parameters
    ----------
    query
        A polars query.
    allow_exceptions
        A tuple of exceptions or an exception that are allowed to be
        raised if their type (not text) matches, we accept that.
    approx
        Whether to use approximate equality for floating point columns
        (and consider NaNs equal as well).
    """
    # Import currently ensures `.legate.collect()` is available
    import legate_dataframe.ldf_polars  # noqa: F401

    exception = None
    try:
        res_polars = query.collect().to_arrow()
    except allow_exceptions as e:
        exception = e
    try:
        res_legate = query.legate.collect().to_arrow()
    except allow_exceptions as e:
        if type(exception) is type(e):
            return  # OK, types match so we accept this.
        if exception is not None:
            raise exception
        raise

    assert_arrow_table_equal(res_legate, res_polars, approx=approx)


def get_empty_series(dtype, nullable: bool) -> Any:
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
    import cudf

    ret = cudf.Series([], dtype=dtype)
    if nullable:
        ret._column.set_mask(np.empty(shape=(0,), dtype="uint8"))
    return ret


def std_dataframe_set() -> List[pa.Table]:
    """Return the standard test set of dataframes

    Used throughout the test suite to check against supported data types

    Returns
    -------
        List of dataframes
    """
    return [
        pa.table({"a": np.arange(10000, dtype="int64")}),
        pa.table(
            {
                "a": np.arange(10000, dtype="int32"),
                "b": np.arange(-10000, 0, dtype="float64"),
                "c": np.resize([True, False], 10000).astype(np.bool_),
            }
        ),
        pa.table({"a": ["a", "bb", "ccc"]}),
        pa.table(
            {
                "a": np.array([], dtype=int),
                "b": np.array([], dtype=float),
            }
        ),
    ]


def gen_random_series(nelem: int, num_nans: int) -> pa.Array:
    rng = np.random.default_rng(42)
    a = rng.random(nelem)
    nans = np.zeros(nelem, dtype=bool)
    nans[rng.choice(a.size, num_nans, replace=False)] = True
    return pa.array(a, mask=nans)


# To replace the above eventually as cudf is removed from tests
def get_pyarrow_column_set(dtypes, nulls=True):
    """Return a set of columns with the given dtypes

    Can be used to test a pytest fixture to generate a set of columns.

    Parameters
    ----------
    dtypes : sequence of dtypes
        The dtypes for the returned columns.
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
        mask = np.random.randint(2, size=len(data), dtype=bool) if nulls else None
        series = pa.array(data, mask=mask).cast(dtype)

        yield pytest.param(series, id=f"col({dtype}, nulls={nulls})")


def guess_available_mem():
    """Function that guesses the available GPU and SYSMEM memory in MiB based
    on the ``LEGATE_CONFIG`` environment variable.
    If the variable is not found or doesn't include ``--fbmem``/``--sysmem``
    returns None for the non-available one.

    Returns a tuple of ``(gpumem, sysmem)``
    """
    config = os.environ.get("LEGATE_CONFIG", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fbmem", type=int, default=None)
    # could probably factor in CPUs, but probably OK in practice.
    parser.add_argument("--sysmem", type=int, default=None)

    args, _ = parser.parse_known_args(config.split())

    ngpus = legate.core.get_machine().count(legate.core.TaskTarget.GPU)

    fbmem = args.fbmem * ngpus if args.fbmem is not None else None
    sysmem = args.sysmem if args.sysmem is not None else None

    return fbmem, sysmem


def get_test_scoping():
    """Return a list of machine scopes for testing with different number of processors.

    The list will contain scopes with 1,2,4,... processors up to the number of
    available processors. If there are GPUs available, it will use those,
    otherwise it will use CPUs.
    """
    # avoid TaskTarget.OMP - sort does not implement this
    runtime = get_legate_runtime()
    n_cpus = runtime.get_machine().count(TaskTarget.CPU)
    n_gpus = runtime.get_machine().count(TaskTarget.GPU)
    target = TaskTarget.GPU if n_gpus > 0 else TaskTarget.CPU
    n_processors = n_gpus if target == TaskTarget.GPU else n_cpus
    i = 1
    scopes_to_test = []
    while i < n_processors:
        scopes_to_test.append(runtime.get_machine().only(target)[:i])
        i *= 2
    scopes_to_test.append(runtime.get_machine().only(target)[:n_processors])
    return scopes_to_test
