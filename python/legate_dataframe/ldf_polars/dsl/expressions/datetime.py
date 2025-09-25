# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.lib import binaryop, timestamps

if TYPE_CHECKING:
    from polars.polars import _expr_nodes as pl_expr
    from typing_extensions import Self

    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["TemporalFunction"]


class TemporalFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `TemporalFunction`."""

        BaseUtcOffset = auto()
        CastTimeUnit = auto()
        Century = auto()
        Combine = auto()
        ConvertTimeZone = auto()
        DSTOffset = auto()
        Date = auto()
        Datetime = auto()
        DatetimeFunction = auto()
        Day = auto()
        Duration = auto()
        Hour = auto()
        IsLeapYear = auto()
        IsoYear = auto()
        Microsecond = auto()
        Millennium = auto()
        Millisecond = auto()
        Minute = auto()
        Month = auto()
        MonthEnd = auto()
        MonthStart = auto()
        Nanosecond = auto()
        OffsetBy = auto()
        OrdinalDay = auto()
        Quarter = auto()
        Replace = auto()
        ReplaceTimeZone = auto()
        Round = auto()
        Second = auto()
        Time = auto()
        TimeStamp = auto()
        ToString = auto()
        TotalDays = auto()
        TotalHours = auto()
        TotalMicroseconds = auto()
        TotalMilliseconds = auto()
        TotalMinutes = auto()
        TotalNanoseconds = auto()
        TotalSeconds = auto()
        Truncate = auto()
        Week = auto()
        WeekDay = auto()
        WithTimeUnit = auto()
        Year = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.TemporalFunction) -> Self:
            """Convert from polars' `TemporalFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "TemporalFunction":
                raise ValueError("TemporalFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")
    _COMPONENT_MAP: ClassVar[dict[Name, str]] = {
        Name.Year: "year",
        Name.Month: "month",
        Name.Day: "day",
        Name.WeekDay: "day_of_week",
        Name.Hour: "hour",
        Name.Minute: "minute",
        Name.Second: "second",
        Name.Millisecond: "millisecond",
        Name.Microsecond: "microsecond",
        Name.Nanosecond: "nanosecond",
    }

    _valid_ops: ClassVar[set[Name]] = {
        *_COMPONENT_MAP.keys(),
        Name.IsLeapYear,
        Name.OrdinalDay,
        Name.MonthStart,
        Name.MonthEnd,
    }

    def __init__(
        self,
        dtype: pa.DataType,
        name: TemporalFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"Temporal function {self.name}")

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        (column,) = columns
        if self.name is TemporalFunction.Name.MonthStart:
            raise NotImplementedError("MonthStart is not implemented")
        if self.name is TemporalFunction.Name.MonthEnd:
            raise NotImplementedError("MonthEnd is not implemented")
        if self.name is TemporalFunction.Name.IsLeapYear:
            raise NotImplementedError("IsLeapYear is not implemented")
        if self.name is TemporalFunction.Name.OrdinalDay:
            raise NotImplementedError("OrdinalDay is not implemented")
        if self.name is TemporalFunction.Name.Microsecond:
            millis = timestamps.extract_timestamp_component(column.obj, "millisecond")
            micros = timestamps.extract_timestamp_component(column.obj, "microsecond")
            millis_as_micros = binaryop.binary_operation(
                millis,
                pa.scalar(1_000, type=pa.int32()),
                "multiply",
                pa.int32(),
            )
            total_micros = binaryop.binary_operation(
                micros,
                millis_as_micros,
                "add",
                pa.int32(),
            )
            return Column(total_micros)
        elif self.name is TemporalFunction.Name.Nanosecond:
            millis = timestamps.extract_timestamp_component(column.obj, "millisecond")
            micros = timestamps.extract_timestamp_component(column.obj, "microsecond")
            nanos = timestamps.extract_timestamp_component(column.obj, "nanosecond")
            millis_as_nanos = binaryop.binary_operation(
                millis,
                pa.scalar(1_000_000, type=pa.int32()),
                "multiply",
                pa.int32(),
            )
            micros_as_nanos = binaryop.binary_operation(
                micros,
                pa.scalar(1_000, type=pa.int32()),
                "multiply",
                pa.int32(),
            )
            total_nanos = binaryop.binary_operation(
                nanos,
                millis_as_nanos,
                "add",
                pa.int32(),
            )
            total_nanos = binaryop.binary_operation(
                total_nanos,
                micros_as_nanos,
                "add",
                pa.int32(),
            )
            return Column(total_nanos)

        return Column(
            timestamps.extract_timestamp_component(
                column.obj,
                self._COMPONENT_MAP[self.name],
            )
        )
