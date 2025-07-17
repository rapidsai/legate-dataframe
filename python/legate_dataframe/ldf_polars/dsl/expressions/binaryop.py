# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""BinaryOp DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pylibcudf as plc
from polars.polars import _expr_nodes as pl_expr

from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.lib.binaryop import binary_operation

if TYPE_CHECKING:
    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["BinOp"]


class BinOp(Expr):
    __slots__ = ("op",)
    _non_child = ("dtype", "op")

    def __init__(
        self,
        dtype: plc.DataType,
        op: str,
        left: Expr,
        right: Expr,
    ) -> None:
        self.dtype = dtype
        if plc.traits.is_boolean(self.dtype):
            # For boolean output types, bitand and bitor implement
            # boolean logic, so translate. bitxor also does, but the
            # default behaviour is correct.
            op = BinOp._BOOL_KLEENE_MAPPING.get(op, op)
        self.op = op
        self.children = (left, right)
        self.is_pointwise = True
        # TODO: Would be nice to have, but I don't want to map via plc to get there...
        # if not plc.binaryop.is_supported_operation(
        #     self.dtype, left.dtype, right.dtype, op
        # ):
        #     raise NotImplementedError(
        #         f"Operation {op.name} not supported "
        #         f"for types {left.dtype.id().name} and {right.dtype.id().name} "
        #         f"with output type {self.dtype.id().name}"
        #     )

    _BOOL_KLEENE_MAPPING: ClassVar[dict[plc.binaryop.BinaryOperator, str]] = {
        # Commented out ones are currently unsupported on the C-side or not clearly mapped
        # due to nullable handling (also below)
        # plc.binaryop.BinaryOperator.BITWISE_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        # plc.binaryop.BinaryOperator.BITWISE_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
        # plc.binaryop.BinaryOperator.LOGICAL_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        # plc.binaryop.BinaryOperator.LOGICAL_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    }

    _MAPPING: ClassVar[dict[pl_expr.Operator, str]] = {
        pl_expr.Operator.Eq: "equal",
        # pl_expr.Operator.EqValidity: "equal",
        pl_expr.Operator.NotEq: "not_equal",
        # pl_expr.Operator.NotEqValidity: "not_equal",
        pl_expr.Operator.Lt: "less",
        pl_expr.Operator.LtEq: "less_equal",
        pl_expr.Operator.Gt: "greater",
        pl_expr.Operator.GtEq: "greater_equal",
        pl_expr.Operator.Plus: "add",
        pl_expr.Operator.Minus: "subtract",
        pl_expr.Operator.Multiply: "multiply",
        pl_expr.Operator.Divide: "divide",
        # pl_expr.Operator.TrueDivide: "true_divide",
        # pl_expr.Operator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
        # pl_expr.Operator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
        pl_expr.Operator.And: "bit_wise_and",
        pl_expr.Operator.Or: "bit_wise_or",
        pl_expr.Operator.Xor: "bit_wise_xor",
        # pl_expr.Operator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
        # pl_expr.Operator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
    }

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        left, right = (child.evaluate(df, context=context) for child in self.children)
        lop = left.obj
        rop = right.obj
        return Column(binary_operation(lop, rop, self.op, self.dtype))
