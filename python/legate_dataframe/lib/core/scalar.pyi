# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numbers

import cudf
import legate.core
import numpy
import pyarrow as pa
from pylibcudf.scalar import Scalar as PylibcudfScalar

ScalarLike = (
    numpy.number
    | numbers.Number
    | cudf.Scalar
    | legate.core.Scalar
    | PylibcudfScalar
    | pa.Scalar
)
