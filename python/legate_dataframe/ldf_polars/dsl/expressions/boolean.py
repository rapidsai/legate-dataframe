# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document BooleanFunction to remove noqa
# ruff: noqa: D101
"""Boolean DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl import expr
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.lib import search, unaryop

if TYPE_CHECKING:
    from cudf_polars.containers import DataFrame
    from polars.polars import _expr_nodes as pl_expr
    from typing_extensions import Self

__all__ = ["BooleanFunction"]


class BooleanFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `BooleanFunction`."""

        All = auto()
        AllHorizontal = auto()
        Any = auto()
        AnyHorizontal = auto()
        IsBetween = auto()  # implemented as binary operation
        IsClose = auto()
        IsDuplicated = auto()
        IsFinite = auto()
        IsFirstDistinct = auto()
        IsIn = auto()
        IsInfinite = auto()
        IsLastDistinct = auto()
        IsNan = auto()
        IsNotNan = auto()
        IsNotNull = auto()
        IsNull = auto()
        IsUnique = auto()
        Not = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.BooleanFunction) -> Self:
            """Convert from polars' `BooleanFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "BooleanFunction":
                raise ValueError("BooleanFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: pa.DataType,
        name: BooleanFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name not in (
            BooleanFunction.Name.All,
            BooleanFunction.Name.Any,
            BooleanFunction.Name.IsDuplicated,
            BooleanFunction.Name.IsFirstDistinct,
            BooleanFunction.Name.IsLastDistinct,
            BooleanFunction.Name.IsUnique,
        )
        if self.name not in {
            BooleanFunction.Name.IsIn,
            BooleanFunction.Name.IsNull,
            BooleanFunction.Name.IsNotNull,
            BooleanFunction.Name.IsNan,
            # BooleanFunction.Name.IsNotNan,  # seems not directly in arrow
            BooleanFunction.Name.Not,
        }:
            raise NotImplementedError(
                f"Boolean function {self.name}"
            )  # pragma: no cover

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is BooleanFunction.Name.IsIn:
            if self.options:  # old versions do not have nans_equal
                (nans_equal,) = self.options
                if nans_equal:
                    # Arrow has this (and more) but libcudf does not currently.
                    raise NotImplementedError(
                        "nans_equal to `is_in` is not implemented"
                    )
            assert len(self.children) == 2
            # TODO(seberg): This is a a bit of a hack, polars wants the literal to be a list one
            # but we force it to the current dtype on the (reconstructed) literal column instead.
            if isinstance(self.children[1], expr.LiteralColumn) and pa.types.is_list(
                self.children[1].dtype
            ):
                haystack_child = self.children[1].reconstruct([])
                haystack_child.dtype = self.children[0].dtype
                haystack = haystack_child.evaluate(df, context=context)
            else:
                haystack = self.children[1].evaluate(df, context=context)

            needles = self.children[0].evaluate(df, context=context)
            # We don't support list type yet, but keep check for now anyway
            if pa.types.is_list(haystack.obj.dtype()):
                raise NotImplementedError(
                    "IsIn with list type not supported (should unwrap)"
                )

            return Column(search.contains(haystack.obj, needles.obj))
        if self.name is BooleanFunction.Name.IsNull:
            col = self.children[0].evaluate(df, context=context)
            return Column(unaryop.unary_operation(col.obj, "is_null"))
        if self.name is BooleanFunction.Name.IsNotNull:
            col = self.children[0].evaluate(df, context=context)
            return Column(unaryop.unary_operation(col.obj, "is_valid"))
        if self.name is BooleanFunction.Name.IsNan:
            col = self.children[0].evaluate(df, context=context)
            return Column(unaryop.unary_operation(col.obj, "is_nan"))
        if self.name is BooleanFunction.Name.Not:
            col = self.children[0].evaluate(df, context=context)
            return Column(unaryop.unary_operation(col.obj, "invert"))
        else:
            raise NotImplementedError(
                f"BooleanFunction {self.name}"
            )  # pragma: no cover; handled by init raising
