# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A column, with some properties."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pylibcudf as plc

from legate_dataframe import LogicalColumn
from legate_dataframe.lib import unaryop

if TYPE_CHECKING:
    from typing_extensions import Self

    from legate_dataframe.ldf_polars.typing import Slice

__all__: list[str] = ["Column"]


class Column:
    """An immutable column with sortedness metadata."""

    obj: LogicalColumn
    # NOTE: cudf_polars stores sortedness, we may need that eventually.
    is_scalar: bool
    # Optional name, only ever set by evaluation of NamedExpr nodes
    # The internal evaluation should not care about the name.
    name: str | None

    def __init__(
        self,
        column: LogicalColumn,
        *,
        name: str | None = None,
    ):
        self.obj = column
        # NOTE: Original assumes len 1 is scalar, may need that relaxed definition?
        self.is_scalar = column.is_scalar()
        self.name = name

    def rename(self, name: str | None, /) -> Self:
        """
        Return a shallow copy with a new name.

        Parameters
        ----------
        name
            New name

        Returns
        -------
        Shallow copy of self with new name set.
        """
        new = self.copy()
        new.name = name
        return new

    def astype(self, dtype: plc.DataType) -> Column:
        """
        Cast the column to as the requested dtype.

        Parameters
        ----------
        dtype
            Datatype to cast to.

        Returns
        -------
        Column of requested type.

        Raises
        ------
        RuntimeError
            If the cast is unsupported.

        Notes
        -----
        This only produces a copy if the requested dtype doesn't match
        the current one.
        """
        if self.obj.type() == dtype:
            return self

        # TODO: Not all casts are supported by direct casts here.  E.g. for
        # string -> numeric we need special calls (or adapt legate-df itself)
        return Column(unaryop.cast(self.obj, dtype))

    def copy(self) -> Self:
        """
        A shallow copy of the column.

        Returns
        -------
        New column sharing data with self.
        """
        return type(self)(
            self.obj,
            name=self.name,
        )

    def mask_nans(self) -> Self:
        """Return a shallow copy of self with nans masked out."""
        raise NotImplementedError("mask_nans not implemented")

    @functools.cached_property
    def nan_count(self) -> int:
        """Return the number of NaN values in the column."""
        raise NotImplementedError("nan_count")

    @property
    def size(self) -> int:
        """Return the size of the column."""
        return self.obj.num_rows()

    @property
    def null_count(self) -> int:
        """Return the number of Null values in the column."""
        raise NotImplementedError(
            "null_count not implemented, it requires blocking also"
        )

    def slice(self, zlice: Slice | None) -> Self:
        """
        Slice a column.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.

        Returns
        -------
        New column (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self.copy()

        # TODO: This is a more important one, due to .head() and .tail()
        raise NotImplementedError("slice not implemented")
