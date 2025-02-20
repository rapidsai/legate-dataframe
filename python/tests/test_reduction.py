# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy
import pytest
from pylibcudf import aggregation

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.reduction import reduce


@pytest.mark.parametrize("agg", ["mean", "max", "min", "sum", "product"])
def test_reduce_simple(agg):
    cupy.random.seed(0)
    cudf_col = cudf.Series(cupy.random.random(size=1000))
    lg_col = LogicalColumn.from_cudf(cudf_col._column)

    cudf_res = getattr(cudf_col, agg)()
    lg_res = reduce(lg_col, getattr(aggregation, agg)(), cudf_res.dtype)

    lg_res_scalar = lg_res.to_cudf_scalar()
    assert lg_res.is_scalar()  # the result should be marked as scalar
    assert lg_res_scalar.is_valid()
    assert lg_res_scalar.value == cudf_res


@pytest.mark.parametrize("agg", ["mean", "max", "sum"])
def test_empty_reduce_simple(agg):
    # Empty aggregations should return null scalars
    cudf_col = cudf.Series([], dtype="float64")
    lg_col = LogicalColumn.from_cudf(cudf_col._column)

    lg_res = reduce(lg_col, getattr(aggregation, agg)(), cudf_col.dtype)

    lg_res_scalar = lg_res.to_cudf_scalar()
    assert lg_res.is_scalar()
    assert not lg_res_scalar.is_valid()


@pytest.mark.parametrize(
    "agg,initial", [("sum", 0.5), ("max", 0), ("min", -1), ("product", 2)]
)
def test_reduce_initial(agg, initial):
    cupy.random.seed(0)
    # Keep values simple to avoid numerical difference:
    cudf_col = cudf.Series([0.5] * 100)
    cudf_col[0] = initial
    # Skip first value, and instead make it the initial:
    lg_col = LogicalColumn.from_cudf(cudf_col[1:]._column)

    cudf_res = getattr(cudf_col, agg)()
    lg_res = reduce(
        lg_col,
        getattr(aggregation, agg)(),
        cudf_res.dtype,
        initial=cudf.Scalar(initial, cudf_col.dtype),
    )

    lg_res_scalar = lg_res.to_cudf_scalar()
    assert lg_res.is_scalar()  # the result should be marked as scalar
    assert lg_res_scalar.is_valid()
    assert lg_res_scalar.value == cudf_res
