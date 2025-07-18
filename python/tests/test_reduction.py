# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.reduction import reduce
from legate_dataframe.testing import assert_matches_polars, gen_random_series


@pytest.mark.parametrize("op", ["mean", "max", "min", "sum", "product"])
def test_reduce_simple(op):
    rng = np.random.RandomState(0)
    array = pa.array(rng.random(size=1000))
    col = LogicalColumn.from_arrow(array)

    result = reduce(col, op, array.type)
    expected = getattr(pa.compute, op)(array)

    assert result.is_scalar()  # the result should be marked as scalar
    assert np.isclose(result.to_arrow()[0].as_py(), expected.as_py())


@pytest.mark.parametrize("op", ["mean", "max", "sum"])
def test_empty_reduce_simple(op):
    # Empty aggregations should return null scalars
    array = pa.array([], type=pa.float64())
    col = LogicalColumn.from_arrow(array)
    result = reduce(col, op, array.type)
    assert result.is_scalar()
    assert result.to_arrow()[0].as_py() is None


@pytest.mark.parametrize(
    "op,initial", [("sum", 0.5), ("max", 0), ("min", -1), ("product", 2)]
)
def test_reduce_initial(op, initial):
    array = pa.array([initial] + [0.5] * 99)
    col = LogicalColumn.from_arrow(array[1:])
    expected = getattr(pa.compute, op)(array)
    result = reduce(col, op, array.type, initial=pa.scalar(initial, array.type))
    assert result.is_scalar()
    assert np.isclose(result.to_arrow()[0].as_py(), expected.as_py())


@pytest.mark.parametrize("op", ["mean", "sum", "min", "max", "first", "last", "count"])
def test_reduce_polars(op):
    pl = pytest.importorskip("polars")

    # A float column with nulls, a float column with nans and an int one
    a = gen_random_series(nelem=1000, num_nans=10)
    arr = np.random.random(1000)
    arr[[100, 200, 300]] = np.nan
    b = pa.array(arr)
    c = pa.array(np.arange(1000))

    q = pl.LazyFrame(
        {"a": pl.from_arrow(a), "b": pl.from_arrow(b), "c": pl.from_arrow(c)}
    )
    q = getattr(q, op)()
    assert_matches_polars(q, approx=True)
