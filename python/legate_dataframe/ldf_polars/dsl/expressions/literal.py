# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Literal DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import pyarrow as pa

from legate_dataframe import LogicalColumn
from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Hashable

    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["Literal", "LiteralColumn"]


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: Any  # Python scalar

    def __init__(self, dtype: pa.DataType, value: Any) -> None:
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        pa_scalar = pa.scalar(self.value, type=self.dtype)

        return Column(LogicalColumn.from_arrow(pa_scalar))

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover


class LiteralColumn(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pa.Array[Any]

    def __init__(self, dtype: pa.DataType, value: pa.Array) -> None:
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype, id(self.value))

    def do_evaluate(
        self,
        df: DataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> Column:
        array = pa.array(self.value)
        return Column(LogicalColumn.from_arrow(array)).astype(self.dtype)

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover
