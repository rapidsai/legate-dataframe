# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from legate_dataframe import LogicalColumn

num = pytest.importorskip("cupynumeric")


def test_column_round_trip():
    a = num.arange(100)
    col = LogicalColumn(a)
    b = num.asarray(col)
    assert num.array_equal(a, b)


def test_column_interop():
    a = num.arange(100)
    col = LogicalColumn(a)
    b = num.add(a, col)
    assert num.array_equal(b, a + a)
