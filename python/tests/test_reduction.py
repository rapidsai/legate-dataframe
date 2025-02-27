# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy
import pytest
from pylibcudf import aggregation

from legate_dataframe import LogicalColumn
from legate_dataframe.lib.reduction import reduce


@pytest.mark.parametrize("agg", ["mean", "max", "min", "sum"])
def test_reduce_simple(agg):
    cupy.random.seed(0)
    cudf_col = cudf.Series(cupy.random.random(size=1000))
    lg_col = LogicalColumn.from_cudf(cudf_col._column)

    lg_res = reduce(lg_col, getattr(aggregation, agg)())
    cudf_res = getattr(cudf_col, agg)()

    lg_res_scalar = lg_res.to_cudf_scalar()
    assert lg_res.scalar()  # the result should be marked as scalar
    assert lg_res.is_valid()
    assert lg_res_scalar.value == cudf_res.value


@pytest.mark.parametrize("agg", ["mean", "max", "min", "sum"])
def test_empty_reduce_simple(agg):
    # Empty aggregations should return null scalars
    cudf_col = cudf.Series([])
    lg_col = LogicalColumn.from_cudf(cudf_col._column)

    lg_res = reduce(lg_col, getattr(aggregation, agg)())

    lg_res_scalar = lg_res.to_cudf_scalar()
    assert lg_res.scalar()
    assert lg_res.is_valid() == cudf_res.is_valid()
    assert lg_res_scalar.value == cudf_res.value
