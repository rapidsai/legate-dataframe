# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Datatype utilities."""

from __future__ import annotations

from functools import cache

import polars as pl
import pyarrow as pa
from pylibcudf.traits import (
    is_floating_point,
    is_integral_not_bool,
    is_numeric_not_bool,
)
from typing_extensions import assert_never

__all__ = [
    "can_cast",
    "downcast_arrow_lists",
    "from_polars",
    "is_order_preserving_cast",
    "to_polars",
]
import pylibcudf as plc
from cudf.utils.dtypes import dtype_to_pylibcudf_type


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


def can_cast(from_: plc.DataType, to: plc.DataType) -> bool:
    """
    Can we cast (via :func:`~.pylibcudf.unary.cast`) between two datatypes.

    Parameters
    ----------
    from_
        Source datatype
    to
        Target datatype

    Returns
    -------
    True if casting is supported, False otherwise
    """
    to_is_empty = to.id() == plc.TypeId.EMPTY
    from_is_empty = from_.id() == plc.TypeId.EMPTY
    has_empty = to_is_empty or from_is_empty
    return (
        (
            from_ == to
            or (
                not has_empty
                and (
                    plc.traits.is_fixed_width(to)
                    and plc.traits.is_fixed_width(from_)
                    and plc.unary.is_supported_cast(from_, to)
                )
            )
        )
        or (
            from_.id() == plc.TypeId.STRING
            and not to_is_empty
            and is_numeric_not_bool(to)
        )
        or (
            to.id() == plc.TypeId.STRING
            and not from_is_empty
            and is_numeric_not_bool(from_)
        )
    )


def is_order_preserving_cast(from_: plc.DataType, to: plc.DataType) -> bool:
    """
    Determine if a cast would preserve the order of the source data.

    Parameters
    ----------
    from_
        Source datatype
    to
        Target datatype

    Returns
    -------
    True if the cast is order-preserving, False otherwise
    """
    if from_.id() == to.id():
        return True

    if is_integral_not_bool(from_) and is_integral_not_bool(to):
        # True if signedness is the same and the target is larger
        if plc.traits.is_unsigned(from_) == plc.traits.is_unsigned(to):
            if plc.types.size_of(to) >= plc.types.size_of(from_):
                return True
        elif (plc.traits.is_unsigned(from_) and not plc.traits.is_unsigned(to)) and (
            plc.types.size_of(to) > plc.types.size_of(from_)
        ):
            # Unsigned to signed is order preserving if target is large enough
            # But signed to unsigned is never order preserving due to negative values
            return True
    elif (
        is_floating_point(from_)
        and is_floating_point(to)
        and (plc.types.size_of(to) >= plc.types.size_of(from_))
    ):
        # True if the target is larger
        return True
    return (is_integral_not_bool(from_) and is_floating_point(to)) or (
        is_floating_point(from_) and is_integral_not_bool(to)
    )


@cache
def from_polars(dtype: pl.DataType) -> plc.DataType:
    """
    Convert a polars datatype to a pylibcudf one.

    Parameters
    ----------
    dtype
        Polars dtype to convert

    Returns
    -------
    Matching pylibcudf DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    if isinstance(dtype, pl.Boolean):
        return plc.DataType(plc.TypeId.BOOL8)
    elif isinstance(dtype, pl.Int8):
        return plc.DataType(plc.TypeId.INT8)
    elif isinstance(dtype, pl.Int16):
        return plc.DataType(plc.TypeId.INT16)
    elif isinstance(dtype, pl.Int32):
        return plc.DataType(plc.TypeId.INT32)
    elif isinstance(dtype, pl.Int64):
        return plc.DataType(plc.TypeId.INT64)
    if isinstance(dtype, pl.UInt8):
        return plc.DataType(plc.TypeId.UINT8)
    elif isinstance(dtype, pl.UInt16):
        return plc.DataType(plc.TypeId.UINT16)
    elif isinstance(dtype, pl.UInt32):
        return plc.DataType(plc.TypeId.UINT32)
    elif isinstance(dtype, pl.UInt64):
        return plc.DataType(plc.TypeId.UINT64)
    elif isinstance(dtype, pl.Float32):
        return plc.DataType(plc.TypeId.FLOAT32)
    elif isinstance(dtype, pl.Float64):
        return plc.DataType(plc.TypeId.FLOAT64)
    elif isinstance(dtype, pl.Date):
        return plc.DataType(plc.TypeId.TIMESTAMP_DAYS)
    elif isinstance(dtype, pl.Time):
        raise NotImplementedError("Time of day dtype not implemented")
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            raise NotImplementedError("Time zone support")
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.Duration):
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.DURATION_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.DURATION_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.String):
        return plc.DataType(plc.TypeId.STRING)
    elif isinstance(dtype, pl.Null):
        # TODO: Hopefully
        return plc.DataType(plc.TypeId.EMPTY)
    elif isinstance(dtype, pl.List):
        # Recurse to catch unsupported inner types
        _ = from_polars(dtype.inner)
        return plc.DataType(plc.TypeId.LIST)
    else:
        raise NotImplementedError(f"{dtype=} conversion not supported")


@cache
def to_polars(dtype) -> pl.DataType:
    """
    Convert pylibcudf typeid to polars.

    Parameters
    ----------
    dtype
        numpy dtype to convert

    Returns
    -------
    Matching polars DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    type_id = dtype_to_pylibcudf_type(dtype).id()

    if type_id == plc.TypeId.BOOL8:
        return pl.Boolean
    elif type_id == plc.TypeId.INT8:
        return pl.Int8
    elif type_id == plc.TypeId.INT16:
        return pl.Int16
    elif type_id == plc.TypeId.INT32:
        return pl.Int32
    elif type_id == plc.TypeId.INT64:
        return pl.Int64
    elif type_id == plc.TypeId.UINT8:
        return pl.UInt8
    elif type_id == plc.TypeId.UINT16:
        return pl.UInt16
    elif type_id == plc.TypeId.UINT32:
        return pl.UInt32
    elif type_id == plc.TypeId.UINT64:
        return pl.UInt64
    elif type_id == plc.TypeId.FLOAT32:
        return pl.Float32
    elif type_id == plc.TypeId.FLOAT64:
        return pl.Float64
    elif type_id == plc.TypeId.TIMESTAMP_DAYS:
        return pl.Date
    elif type_id == plc.TypeId.STRING:
        return pl.String

    raise NotImplementedError("Type translation not implemented")
