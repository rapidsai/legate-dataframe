# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
"""DSL nodes for unary operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import cudf  # eventually, we should not need this
import pylibcudf as plc

from legate_dataframe import LogicalColumn
from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.ldf_polars.utils import dtypes
from legate_dataframe.lib import replace, unaryop

if TYPE_CHECKING:

    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["Cast", "Len", "UnaryFunction"]


class Cast(Expr):
    """Class representing a cast of an expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, value: Expr) -> None:
        self.dtype = dtype
        self.children = (value,)
        self.is_pointwise = True
        if not dtypes.can_cast(value.dtype, self.dtype):
            raise NotImplementedError(
                f"Can't cast {value.dtype.id().name} to {self.dtype.id().name}"
            )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        return column.astype(self.dtype)


class Len(Expr):
    """Class representing the length of an expression."""

    def __init__(self, dtype: plc.DataType) -> None:
        self.dtype = dtype
        self.children = ()
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            LogicalColumn.from_cudf(cudf.Scalar(df.num_rows, dtype=self.dtype))
        )


class UnaryFunction(Expr):
    """Class representing unary functions of an expression."""

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    # Note: log, and pow are handled via translation to binops
    _OP_MAPPING: ClassVar[dict[str, plc.unary.UnaryOperator]] = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "arcsin": "asin",
        "arccos": "acos",
        "arctan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "arcsinh": "asinh",
        "arccosh": "acosh",
        "arctanh": "atanh",
        "exp": "exp",
        "sqrt": "sqrt",
        # "cbrt": "cbrt",
        "ceil": "ceil",
        "floor": "floor",
        "abs": "abs",
        "bit_invert": "bit_not",
        "not": "bit_not",
        # "negate": plc.unary.UnaryOperator.NEGATE,  Needs RAPIDS 25.06 (or so)
    }
    _supported_misc_fns = frozenset(
        {
            "drop_nulls",
            "fill_null",
            "mask_nans",
            "round",
            "set_sorted",
            "unique",
        }
    )
    _supported_cum_aggs = frozenset(
        {
            "cum_min",
            "cum_max",
            "cum_prod",
            "cum_sum",
        }
    )
    _supported_fns = frozenset().union(
        _supported_misc_fns, _supported_cum_aggs, _OP_MAPPING.keys()
    )

    def __init__(
        self, dtype: plc.DataType, name: str, options: tuple[Any, ...], *children: Expr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.children = children
        self.is_pointwise = self.name not in (
            "cum_min",
            "cum_max",
            "cum_prod",
            "cum_sum",
            "drop_nulls",
            "unique",
        )

        if self.name not in UnaryFunction._supported_fns:
            raise NotImplementedError(f"Unary function {name=}")
        if self.name in UnaryFunction._supported_cum_aggs:
            (reverse,) = self.options
            if reverse:
                raise NotImplementedError(
                    "reverse=True is not supported for cumulative aggregations"
                )

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # TODO: The following branches are not implemented:
        # - mask_nans
        # - round
        # - unique
        # - set_sorted
        # - cum_min
        # - cum_max
        # - cum_prod
        # - cum_sum
        if self.name == "drop_nulls":
            # Could implement it via apply boolean mask, but probably better to do it explicitly
            raise NotImplementedError(
                "drop_nulls not implemented (but should prioritize this)"
            )
        elif self.name == "fill_null":
            column = self.children[0].evaluate(df, context=context)
            arg = self.children[1].evaluate(df, context=context).obj
            # TODO: may need to cast, at least for polars>1.28 for scalar columns
            if not arg.is_scalar():
                raise NotImplementedError("fill_null with non-scalar not implemented")

            return Column(replace.replace_nulls(column.obj, arg))
        elif self.name in self._OP_MAPPING:
            column = self.children[0].evaluate(df, context=context)
            if column.obj.type().id() != self.dtype.id():
                arg = unaryop.cast(column.obj, self.dtype)
            else:
                arg = column.obj
            return Column(unaryop.unary_operation(arg, self._OP_MAPPING[self.name]))
        raise NotImplementedError(
            f"Unimplemented unary function {self.name=}"
        )  # pragma: no cover; init trips first
