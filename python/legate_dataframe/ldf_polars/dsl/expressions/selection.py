# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for selection operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from legate_dataframe import LogicalTable
from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.lib import stream_compaction

if TYPE_CHECKING:
    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["Filter", "Gather"]


class Gather(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr) -> None:
        self.dtype = dtype
        self.children = (values, indices)
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        raise NotImplementedError("gather not implemented")


class Filter(Expr):
    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: plc.DataType, values: Expr, indices: Expr):
        self.dtype = dtype
        self.children = (values, indices)
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        values, mask = (child.evaluate(df, context=context) for child in self.children)
        table = stream_compaction.apply_boolean_mask(
            LogicalTable([values.obj], ["column"]), mask.obj
        )
        return Column(table["column"])
