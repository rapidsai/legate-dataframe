# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""
DSL nodes for the polars expression language.

An expression node is a function, `DataFrame -> Column`.

The evaluation context is provided by a LogicalPlan node, and can
affect the evaluation rule as well as providing the dataframe input.
In particular, the interpretation of the expression language in a
`GroupBy` node is groupwise, rather than whole frame.
"""

from __future__ import annotations

from legate_dataframe.ldf_polars.dsl.expressions.aggregation import Agg
from legate_dataframe.ldf_polars.dsl.expressions.base import (
    AggInfo,
    Col,
    ColRef,
    ErrorExpr,
    Expr,
    NamedExpr,
)
from legate_dataframe.ldf_polars.dsl.expressions.binaryop import BinOp
from legate_dataframe.ldf_polars.dsl.expressions.boolean import BooleanFunction
from legate_dataframe.ldf_polars.dsl.expressions.datetime import TemporalFunction
from legate_dataframe.ldf_polars.dsl.expressions.literal import Literal, LiteralColumn

# from legate_dataframe.ldf_polars.dsl.expressions.rolling import GroupedRollingWindow, RollingWindow
from legate_dataframe.ldf_polars.dsl.expressions.selection import Filter
from legate_dataframe.ldf_polars.dsl.expressions.slicing import Slice

# from legate_dataframe.ldf_polars.dsl.expressions.sorting import Sort, SortBy
# from legate_dataframe.ldf_polars.dsl.expressions.string import StringFunction
from legate_dataframe.ldf_polars.dsl.expressions.ternary import Ternary
from legate_dataframe.ldf_polars.dsl.expressions.unary import Cast, Len, UnaryFunction

__all__ = [
    "Agg",
    "AggInfo",
    "BinOp",
    "BooleanFunction",
    "Cast",
    "Col",
    "ColRef",
    "ErrorExpr",
    "Expr",
    "Filter",
    # "Gather",
    # "GroupedRollingWindow",
    "Len",
    "Literal",
    "LiteralColumn",
    "NamedExpr",
    # "RollingWindow",
    "Slice",
    # "Sort",
    # "SortBy",
    # "StringFunction",
    "TemporalFunction",
    "Ternary",
    "UnaryFunction",
]
