# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import cudf
import legate.core
import pyarrow as pa

from pylibcudf.scalar cimport Scalar as PylibcudfScalar

from legate_dataframe.lib.core.column cimport LogicalColumn


cdef LogicalColumn cpp_scalar_col_from_python(scalar: ScalarLike):
    """Convert or ensure any supported Python scalar is a scalar column.

    .. note::
        This is a helper, but we may want to force users to create the
        scalar column themselves at some point.

    Parameters
    ----------
        A legate scalar, pylibcudf Scalar, or object convertible to a cudf scalar.

    Returns
    -------
        Scalar argument
    """
    cdef PylibcudfScalar cudf_scalar

    if isinstance(scalar, LogicalColumn):
        if not scalar.is_scalar():
            raise ValueError(
                "expected a scalar logical column, but column is not scalar.")
        return <LogicalColumn>scalar

    # TODO: it would be good to provide a direct conversion from
    #       `legate.core.Scalar`.
    if isinstance(scalar, legate.core.Scalar):
        scalar = scalar.value()

    if isinstance(scalar, pa.Scalar):
        return LogicalColumn.from_arrow(scalar)

    # NOTE: Converting to a cudf scalar isn't really ideal, as we copy
    #       to the device, just to copy it back again to get a legate one.
    if isinstance(scalar, PylibcudfScalar) :
        cudf_scalar = <PylibcudfScalar>scalar
    else:
        cudf_scalar = <PylibcudfScalar>(cudf.Scalar(scalar).device_value)
    return LogicalColumn.from_cudf(cudf_scalar)
