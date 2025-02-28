# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numbers

import cudf
import legate.core
import numpy
from cudf._lib.scalar import DeviceScalar

ScalarLike = (
    numpy.number | numbers.Number | cudf.Scalar | legate.core.Scalar | DeviceScalar
)
