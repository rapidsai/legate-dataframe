# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference

from cudf._lib.scalar cimport DeviceScalar

import numbers

import cudf
import legate.core
import numpy

ScalarLike = numpy.number | numbers.Number | cudf.Scalar | legate.core.Scalar


cdef cpp_ScalarArg cpp_scalar_arg_from_python(scalar: ScalarLike):
    """Convert any supported Python scalar to a scalar task argument

    Parameters
    ----------
        A legate scalar or object convertible to a scalar argument.

    Returns
    -------
        Scalar argument
    """

    # TODO: use the raw handle of `legate.core.Scalar`. For some reason using
    #       the raw handle results in a segmentation faults thus for now, we
    #       just convert the scalar to a Python value.
    #       See <https://github.com/rapidsai/legate-dataframe/issues/109>
    if isinstance(scalar, legate.core.Scalar):
        scalar = scalar.value()

    # NOTE: Converting to a cudf scalar isn't really ideal, as we copy
    #       to the device, just to copy it back again to get a legate one.
    cdef DeviceScalar cudf_scalar = <DeviceScalar>(cudf.Scalar(scalar).device_value)
    return cpp_ScalarArg(dereference(cudf_scalar.get_raw_ptr()))
