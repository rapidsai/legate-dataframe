# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

from legate_dataframe import LogicalColumn

example_arrays = [
    pa.array([True, False, True], type=pa.bool_()),
    pa.array([1, 2, 3], type=pa.int32()),
    pa.array([1, 2, 3], type=pa.int64()),
    pa.array([1, 2, 3], type=pa.uint32()),
    pa.array([1, 2, 3], type=pa.uint64()),
    pa.array([1, 2, 3], type=pa.float32()),
    pa.array([1, 2, 3], type=pa.float64()),
    pa.array(["1", "2", "3"], type=pa.string()),
    pa.array(["1", "2", "3"], type=pa.large_string()),
    # Times:
    pa.array([1, 2, 3], type=pa.timestamp("s")),
    pa.array([1, 2, 3], type=pa.timestamp("ms")),
    pa.array([1, 2, 3], type=pa.timestamp("us")),
    pa.array([1, 2, 3], type=pa.timestamp("ns")),
    pa.array([1, 2, 3], type=pa.duration("s")),
    pa.array([1, 2, 3], type=pa.duration("ms")),
    pa.array([1, 2, 3], type=pa.duration("us")),
    pa.array([1, 2, 3], type=pa.duration("ns")),
]


@pytest.mark.parametrize("array", example_arrays)
def test_column_round_trip(array):
    expected = array
    if array.type == pa.large_string():
        # If the input is a large string, it's OK to round-trp to string
        expected = pa.array(expected, type=pa.string())
    assert expected == LogicalColumn.from_arrow(array).to_arrow()

    array = pa.array(array, mask=[True, False, True])
    expected = pa.array(expected, mask=[True, False, True])
    assert expected == LogicalColumn.from_arrow(array).to_arrow()


@pytest.mark.parametrize("array", example_arrays)
def test_scalar_round_trip(array):
    scalar = array[0]
    # to_arrow() returns an array (but we do preserve the scalar info)
    expected = array[:1]
    if array.type == pa.large_string():
        # If the input is a large string, it's OK to round-trp to string
        expected = pa.array(expected, type=pa.string())

    col = LogicalColumn.from_arrow(scalar)
    assert col.is_scalar()
    assert expected == col.to_arrow()
