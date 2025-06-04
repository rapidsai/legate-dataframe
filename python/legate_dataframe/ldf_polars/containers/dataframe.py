# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A dataframe, with some properties."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, cast

import polars as pl
import pylibcudf as plc

from legate_dataframe import LogicalTable
from legate_dataframe.ldf_polars.containers import Column
from legate_dataframe.lib import stream_compaction

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    from typing_extensions import Any, Self

    from legate_dataframe.ldf_polars.typing import Slice


__all__: list[str] = ["DataFrame"]


# Pacify the type checker. DataFrame init asserts that all the columns
# have a string name, so let's narrow the type.
class NamedColumn(Column):
    name: str


class DataFrame:
    """A representation of a dataframe."""

    column_map: dict[str, Column]
    table: LogicalTable
    columns: list[NamedColumn]

    def __init__(self, columns: Iterable[Column]) -> None:
        columns = list(columns)
        if any(c.name is None for c in columns):
            raise ValueError("All columns must have a name")
        self.columns = [cast(NamedColumn, c) for c in columns]
        self.column_map = {c.name: c for c in self.columns}
        self.table = LogicalTable(
            [c.obj for c in self.columns], [c.name for c in self.columns]
        )

    def copy(self) -> Self:
        """Return a shallow copy of self."""
        return type(self)(c.copy() for c in self.columns)

    def to_polars(self) -> pl.DataFrame:
        """Convert to a polars DataFrame."""
        # Unlike cudf_polars, we don't want to do this implicitly anyway...
        raise NotImplementedError("Conversion to polars not implemented.")

    @cached_property
    def column_names_set(self) -> frozenset[str]:
        """Return the column names as a set."""
        return frozenset(self.column_map)

    @cached_property
    def column_names(self) -> list[str]:
        """Return a list of the column names."""
        return list(self.column_map)

    @cached_property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.column_map)

    @cached_property
    def num_rows(self) -> int:
        """Number of rows."""
        return self.table.num_rows() if self.column_map else 0

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> Self:
        # Unlike cudf_polars, we don't want to do this implicitly anyway...
        raise NotImplementedError("Can't convert from polars, yet.")

    @classmethod
    def from_table(cls, table: plc.Table) -> Self:
        """
        Create from a legate-dataframe logical table.

        Parameters
        ----------
        table
            LogicalTable to obtain columns from

        Returns
        -------
        New dataframe sharing data with the input table.

        Raises
        ------
        ValueError
            If the number of provided names does not match the
            number of columns in the table.
        """
        names = table.get_column_names()
        return cls([Column(table[name], name=name) for name in names])

    def with_columns(self, columns: Iterable[Column], *, replace_only=False) -> Self:
        """
        Return a new dataframe with extra columns.

        Parameters
        ----------
        columns
            Columns to add
        replace_only
            If true, then only replacements are allowed (matching by name).

        Returns
        -------
        New dataframe

        Notes
        -----
        If column names overlap, newer names replace older ones, and
        appear in the same order as the original frame.
        """
        new = {c.name: c for c in columns}
        if replace_only and not self.column_names_set.issuperset(new.keys()):
            raise ValueError("Cannot replace with non-existing names")
        return type(self)((self.column_map | new).values())

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(column for column in self.columns if column.name not in names)

    def select(self, names: Sequence[str] | Mapping[str, Any]) -> Self:
        """Select columns by name returning DataFrame."""
        try:
            return type(self)(self.column_map[name] for name in names)
        except KeyError as e:
            raise ValueError("Can't select missing names") from e

    def rename_columns(self, mapping: Mapping[str, str]) -> Self:
        """Rename some columns."""
        return type(self)(c.rename(mapping.get(c.name, c.name)) for c in self.columns)

    def select_columns(self, names: Set[str]) -> list[Column]:
        """Select columns by name."""
        return [c for c in self.columns if c.name in names]

    def filter(self, mask: Column) -> Self:
        """Return a filtered table given a mask."""
        table = stream_compaction.apply_boolean_mask(self.table, mask.obj)
        return type(self).from_table(table)

    def slice(self, zlice: Slice | None) -> Self:
        """
        Slice a dataframe.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.

        Returns
        -------
        New dataframe (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self.copy()

        # TODO: This is a more important one, due to .head() and .tail()
        raise NotImplementedError("slice not implemented")
