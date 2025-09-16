# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numbers

import legate.core
import numpy
import pyarrow as pa

ScalarLike = numpy.number | numbers.Number | legate.core.Scalar | pa.Scalar
