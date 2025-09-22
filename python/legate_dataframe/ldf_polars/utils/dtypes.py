# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Datatype utilities."""

from __future__ import annotations

from functools import cache

import polars as pl
import pyarrow as pa

__all__ = [
    "downcast_arrow_lists",
    "from_polars",
    "to_polars",
]


def downcast_arrow_lists(typ: pa.DataType) -> pa.DataType:
    """
    Sanitize an arrow datatype from polars.

    Parameters
    ----------
    typ
        Arrow type to sanitize

    Returns
    -------
    Sanitized arrow type

    Notes
    -----
    As well as arrow ``ListType``s, polars can produce
    ``LargeListType``s and ``FixedSizeListType``s, these are not
    currently handled by libcudf, so we attempt to cast them all into
    normal ``ListType``s on the arrow side before consuming the arrow
    data.
    """
    if isinstance(typ, pa.LargeListType):
        return pa.list_(downcast_arrow_lists(typ.value_type))
    # We don't have to worry about diving into struct types for now
    # since those are always NotImplemented before we get here.
    assert not isinstance(typ, pa.StructType)
    return typ


@cache
def from_polars(dtype: pl.DataType) -> pa.DataType:
    """
    Convert a polars datatype to pyarrow.

    Parameters
    ----------
    dtype
        Polars dtype to convert

    Returns
    -------
    Matching pyarrow DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    scalar = pl.Scalar(None, dtype)
    return scalar.to_arrow().type


@cache
def to_polars(dtype) -> pl.DataType:
    """
    Convert arrow type to polars.

    Parameters
    ----------
    dtype
        arrow dtype to convert

    Returns
    -------
    Matching polars DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    scalar = pa.scalar(None, type=dtype)
    return pl.from_arrow(scalar).dtype
