# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import numbers

import cudf
import legate.core
import numpy

from cudf._lib.scalar cimport DeviceScalar

from legate_dataframe.lib.core.column cimport LogicalColumn

ScalarLike = (
    numpy.number | numbers.Number | cudf.Scalar | legate.core.Scalar | DeviceScalar
)


cdef LogicalColumn cpp_scalar_col_from_python(scalar: ScalarLike):
    """Convert any supported Python scalar to a scalar column.

    .. note::
        This is a helper, but we may want to force users to create the
        scalar column themselves at some point.

    Parameters
    ----------
        A legate scalar, cudf DeviceScalar, or object convertible to a cudf scalar.

    Returns
    -------
        Scalar argument
    """
    cdef DeviceScalar cudf_scalar
    # TODO: it would be good to provide a direct conversion from
    #       `legate.core.Scalar`.
    if isinstance(scalar, legate.core.Scalar):
        scalar = scalar.value()

    # NOTE: Converting to a cudf scalar isn't really ideal, as we copy
    #       to the device, just to copy it back again to get a legate one.
    if isinstance(scalar, DeviceScalar) :
        cudf_scalar = <DeviceScalar>scalar
    else:
        cudf_scalar = <DeviceScalar>(cudf.Scalar(scalar).device_value)
    return LogicalColumn.from_cudf(cudf_scalar)
