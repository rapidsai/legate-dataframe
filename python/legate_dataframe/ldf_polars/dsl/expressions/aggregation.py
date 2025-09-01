# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for aggregations."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute
import pylibcudf as plc

from legate_dataframe import LogicalColumn
from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.lib import reduction

if TYPE_CHECKING:
    from legate_dataframe.containers import DataFrame

__all__ = ["Agg"]


class Agg(Expr):
    __slots__ = ("name", "op", "options", "request")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self, dtype: plc.DataType, name: str, options: Any, *children: Expr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.is_pointwise = False
        self.children = children

        if name == "first":
            self.op = self._first
        elif name == "last":
            self.op = self._last
        elif name in {"mean", "sum"}:
            self.op = functools.partial(self._reduce, request=name)
        elif name in {"min", "max"}:
            # Note: Unlike (py)libcudf we ignore NaNs inside the reduction code itself.
            if options:
                raise NotImplementedError("Min/max always propagate nans currently.")
            self.op = functools.partial(self._reduce, request=name)
        elif name == "count":
            self.op = functools.partial(self._count, include_nulls=options)
        elif name == "n_unique":
            # request=("count_distinct", pa.compute.CountOptions()) not supported yet
            self.op = self._n_unique
        else:
            raise NotImplementedError(f"Unsupported aggregation {name=}.")

    def _reduce(self, column: Column, *, request: str) -> Column:
        return Column(
            reduction.reduce(column.obj, request, self.dtype),
            name=column.name,
        )

    def _n_unique(self, column: Column) -> Column:
        raise NotImplementedError("n_unique not supported yet in normal reduce")

    def _count(self, column: Column, *, include_nulls: bool) -> Column:
        if include_nulls:
            return Column(
                LogicalColumn.from_arrow(pa.scalar(column.size, type=pa.int64()))
            )
        return Column(
            reduction.reduce(column.obj, "count_valid", "int64"),
            name=column.name,
        )

    def _first(self, column: Column) -> Column:
        return column.slice((0, 1))

    def _last(self, column: Column) -> Column:
        return column.slice((-1, None))

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(
                f"Agg in context {context}"
            )  # pragma: no cover; unreachable

        # Aggregations like quantiles may have additional children that were
        # preprocessed into pylibcudf requests.
        in_col = self.children[0].evaluate(df, context=context)
        return self.op(in_col)

    @property
    def agg_request(self):
        if self.name == "n_unique":
            return ("count_distinct", pa.compute.CountOptions(mode="all"))
        return self.name
