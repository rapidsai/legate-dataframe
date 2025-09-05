# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document StringFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for string operations."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.ldf_polars.dsl.expressions.base import ExecutionContext, Expr
from legate_dataframe.ldf_polars.dsl.expressions.literal import Literal
from legate_dataframe.lib import strings

if TYPE_CHECKING:
    from polars.polars import _expr_nodes as pl_expr
    from typing_extensions import Self

    from legate_dataframe.ldf_polars.containers import DataFrame

__all__ = ["StringFunction"]


class StringFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `StringFunction`."""

        Base64Decode = auto()
        Base64Encode = auto()
        ConcatHorizontal = auto()
        ConcatVertical = auto()
        Contains = auto()
        ContainsAny = auto()
        CountMatches = auto()
        EndsWith = auto()
        EscapeRegex = auto()
        Extract = auto()
        ExtractAll = auto()
        ExtractGroups = auto()
        Find = auto()
        Head = auto()
        HexDecode = auto()
        HexEncode = auto()
        JsonDecode = auto()
        JsonPathMatch = auto()
        LenBytes = auto()
        LenChars = auto()
        Lowercase = auto()
        Normalize = auto()
        PadEnd = auto()
        PadStart = auto()
        Replace = auto()
        ReplaceMany = auto()
        Reverse = auto()
        Slice = auto()
        Split = auto()
        SplitExact = auto()
        SplitN = auto()
        StartsWith = auto()
        StripChars = auto()
        StripCharsEnd = auto()
        StripCharsStart = auto()
        StripPrefix = auto()
        StripSuffix = auto()
        Strptime = auto()
        Tail = auto()
        Titlecase = auto()
        ToDecimal = auto()
        ToInteger = auto()
        Uppercase = auto()
        ZFill = auto()

        @classmethod
        def from_polars(cls, obj: pl_expr.StringFunction) -> Self:
            """Convert from polars' `StringFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "StringFunction":
                raise ValueError("StringFunction required")
            return getattr(cls, name)

    _valid_ops: ClassVar[set[Name]] = {
        Name.Contains,
        Name.EndsWith,
        Name.StartsWith,
    }
    __slots__ = ("_pattern", "name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: plc.DataType,
        name: StringFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name != StringFunction.Name.ConcatVertical
        self._validate_input()

    def _validate_input(self) -> None:
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"String function {self.name!r}")
        if self.name is StringFunction.Name.Contains:
            literal, strict = self.options
            if not literal:
                if not strict:
                    raise NotImplementedError(
                        f"{strict=} is not supported for regex contains"
                    )
                if not isinstance(self.children[1], Literal):
                    raise NotImplementedError(
                        "Regex contains only supports a scalar pattern"
                    )
                # pattern = self.children[1].value
                # self._regex_program = self._create_regex_program(pattern)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        column_expr, pattern_expr = self.children
        if not isinstance(pattern_expr, Literal):
            raise NotImplementedError(
                "starts_with, ends_with, and contains only supports a literal prefix"
            )
        else:
            pattern = pattern_expr.value
        column = column_expr.evaluate(df, context=context)

        if self.name is StringFunction.Name.Contains:
            literal, strict = self.options
            assert strict or literal  # should be checked already
            if literal:
                return Column(strings.match("match_substring", column.obj, pattern))
            else:
                return Column(
                    strings.match("match_substring_regex", column.obj, pattern),
                )
        elif self.name is StringFunction.Name.EndsWith:
            return Column(strings.match("ends_with", column.obj, pattern))
        elif self.name is StringFunction.Name.StartsWith:
            return Column(strings.match("starts_with", column.obj, pattern))

        raise NotImplementedError(
            f"StringFunction {self.name}"
        )  # pragma: no cover; handled by init raising
