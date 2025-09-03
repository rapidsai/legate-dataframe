# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
DSL nodes for the LogicalPlan of polars.

An IR node is either a source, normal, or a sink. Respectively they
can be considered as functions:

- source: `IO () -> DataFrame`
- normal: `DataFrame -> DataFrame`
- sink: `DataFrame -> IO ()`
"""

from __future__ import annotations

import json
import random
import time
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import legate_dataframe.ldf_polars.dsl.expr as expr
from legate_dataframe import LogicalTable
from legate_dataframe.ldf_polars.containers import Column, DataFrame
from legate_dataframe.ldf_polars.dsl.nodebase import Node
from legate_dataframe.ldf_polars.utils.versions import POLARS_VERSION_LT_128
from legate_dataframe.lib import copying, csv, groupby_aggregation, join, parquet, sort

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, MutableMapping, Sequence
    from typing import Literal

    from cudf_polars.typing import Schema
    from cudf_polars.typing import Slice as Zlice
    from cudf_polars.utils.config import ConfigOptions
    from cudf_polars.utils.timer import Timer


__all__ = [
    "IR",
    "Cache",
    # "ConditionalJoin",
    "DataFrameScan",
    # "Distinct",
    "ErrorNode",
    "Filter",
    # "GroupBy",
    # "HConcat",
    "HStack",
    "Join",
    # "MapFunction",
    "Projection",
    "PythonScan",
    "Scan",
    "Select",
    "Slice",
    "Sort",
    "Union",
]


def broadcast(*columns: Column, target_length: int | None = None) -> list[Column]:
    """
    Broadcast a sequence of columns to a common length.

    Parameters
    ----------
    columns
        Columns to broadcast.
    target_length
        Optional length to broadcast to. If not provided, uses the
        non-unit length of existing columns.

    Returns
    -------
    List of broadcasted columns all of the same length.

    Raises
    ------
    RuntimeError
        If broadcasting is not possible.

    Notes
    -----
    In evaluation of a set of expressions, polars type-puns length-1
    columns with scalars. When we insert these into a DataFrame
    object, we need to ensure they are of equal length. This function
    takes some columns, some of which may be length-1 and ensures that
    all length-1 columns are broadcast to the length of the others.

    Broadcasting is only possible if the set of lengths of the input
    columns is a subset of ``{1, n}`` for some (fixed) ``n``. If
    ``target_length`` is provided and not all columns are length-1
    (i.e. ``n != 1``), then ``target_length`` must be equal to ``n``.
    """
    if len(columns) == 0:
        return []
    lengths: set[int] = {column.size for column in columns}
    if lengths == {1}:
        if target_length is None:
            return list(columns)
        nrows = target_length
    else:
        try:
            (nrows,) = lengths.difference([1])
        except ValueError as e:
            raise RuntimeError("Mismatching column lengths") from e
        if target_length is not None and nrows != target_length:
            raise RuntimeError(
                f"Cannot broadcast columns of length {nrows=} to {target_length=}"
            )

    # TODO: Need to implement a function to get a repeated column
    def broadcast_func(col, nrows):
        raise NotImplementedError("broadcast not implemented")

    return [
        column if column.size == nrows else broadcast_func(column, nrows)
        for column in columns
    ]


class IR(Node["IR"]):
    """Abstract plan node, representing an unevaluated dataframe."""

    __slots__ = ("_non_child_args", "schema")
    # This annotation is needed because of https://github.com/python/mypy/issues/17981
    _non_child: ClassVar[tuple[str, ...]] = ("schema",)
    # Concrete classes should set this up with the arguments that will
    # be passed to do_evaluate.
    _non_child_args: tuple[Any, ...]
    schema: Schema
    """Mapping from column names to their data types."""

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of node, treating schema dictionary.

        Since the schema is a dictionary, even though it is morally
        immutable, it is not hashable. We therefore convert it to
        tuples for hashing purposes.
        """
        # Schema is the first constructor argument
        args = self._ctor_arguments(self.children)[1:]
        schema_hash = tuple(self.schema.items())
        return (type(self), schema_hash, args)

    # Hacky to avoid type-checking issues, just advertise the
    # signature. Both mypy and pyright complain if we have an abstract
    # method that takes arbitrary *args, but the subclasses have
    # tighter signatures. This complaint is correct because the
    # subclass is not Liskov-substitutable for the superclass.
    # However, we know do_evaluate will only be called with the
    # correct arguments by "construction".
    do_evaluate: Callable[..., DataFrame]
    """
    Evaluate the node (given its evaluated children), and return a dataframe.

    Parameters
    ----------
    args
        Non child arguments followed by any evaluated dataframe inputs.

    Returns
    -------
    DataFrame (on device) representing the evaluation of this plan
    node.

    Raises
    ------
    NotImplementedError
        If evaluation fails. Ideally this should not occur, since the
        translation phase should fail earlier.
    """

    def evaluate(
        self, *, cache: MutableMapping[int, DataFrame], timer: Timer | None
    ) -> DataFrame:
        """
        Evaluate the node (recursively) and return a dataframe.

        Parameters
        ----------
        cache
            Mapping from cached node ids to constructed DataFrames.
            Used to implement evaluation of the `Cache` node.
        timer
            If not None, a Timer object to record timings for the
            evaluation of the node.

        Notes
        -----
        Prefer not to override this method. Instead implement
        :meth:`do_evaluate` which doesn't encode a recursion scheme
        and just assumes already evaluated inputs.

        Returns
        -------
        DataFrame (on device) representing the evaluation of this plan
        node (and its children).

        Raises
        ------
        NotImplementedError
            If evaluation fails. Ideally this should not occur, since the
            translation phase should fail earlier.
        """
        children = [child.evaluate(cache=cache, timer=timer) for child in self.children]
        if timer is not None:
            start = time.monotonic_ns()
            result = self.do_evaluate(*self._non_child_args, *children)
            end = time.monotonic_ns()
            # TODO: Set better names on each class object.
            timer.store(start, end, type(self).__name__)
            return result
        else:
            return self.do_evaluate(*self._non_child_args, *children)


class ErrorNode(IR):
    """Represents an error translating the IR."""

    __slots__ = ("error",)
    _non_child = (
        "schema",
        "error",
    )
    error: str
    """The error."""

    def __init__(self, schema: Schema, error: str):
        self.schema = schema
        self.error = error
        self.children = ()


class PythonScan(IR):
    """Representation of input from a python function."""

    __slots__ = ("options", "predicate")
    _non_child = ("schema", "options", "predicate")
    options: Any
    """Arbitrary options."""
    predicate: expr.NamedExpr | None
    """Filter to apply to the constructed dataframe before returning it."""

    def __init__(self, schema: Schema, options: Any, predicate: expr.NamedExpr | None):
        self.schema = schema
        self.options = options
        self.predicate = predicate
        self._non_child_args = (schema, options, predicate)
        self.children = ()
        # NOTE: supporting this to support a `.lazy()` method on LogicalTable.

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        info: tuple,
        predicate: expr.NamedExpr | None,
        *args,
        **kwargs,
    ):
        if predicate is not None:
            raise NotImplementedError

        if len(args) > 0 or len(kwargs) > 0:
            raise ValueError(f"Unhandled args: {args=} {kwargs=}")

        # TODO: Need to see if any of this matters more
        function, cols, io_plugin_str, None_ = info
        assert None_ is None
        assert io_plugin_str == "io_plugin"

        legate_table = function(None, None, None, None)
        if not isinstance(legate_table, LogicalTable):
            raise NotImplementedError(
                "Function did not return legate table:", type(legate_table)
            )

        if cols is not None:
            legate_table = LogicalTable([legate_table[c] for c in cols], cols)
        return DataFrame.from_table(legate_table)


class Scan(IR):
    """Input from files."""

    __slots__ = (
        "cloud_options",
        "include_file_paths",
        "n_rows",
        "parquet_options",
        "paths",
        "predicate",
        "reader_options",
        "row_index",
        "skip_rows",
        "typ",
        "with_columns",
    )
    _non_child = (
        "schema",
        "typ",
        "reader_options",
        "cloud_options",
        "paths",
        "with_columns",
        "skip_rows",
        "n_rows",
        "row_index",
        "include_file_paths",
        "predicate",
        "parquet_options",
    )
    typ: str
    """What type of file are we reading? Parquet, CSV, etc..."""
    reader_options: dict[str, Any]
    """Reader-specific options, as dictionary."""
    cloud_options: dict[str, Any] | None
    """Cloud-related authentication options, currently ignored."""
    paths: list[str]
    """List of paths to read from."""
    with_columns: list[str] | None
    """Projected columns to return."""
    skip_rows: int
    """Rows to skip at the start when reading."""
    n_rows: int
    """Number of rows to read after skipping."""
    row_index: tuple[str, int] | None
    """If not None add an integer index column of the given name."""
    include_file_paths: str | None
    """Include the path of the source file(s) as a column with this name."""
    predicate: expr.NamedExpr | None
    """Mask to apply to the read dataframe."""
    parquet_options: None
    """Parquet-specific options."""

    PARQUET_DEFAULT_CHUNK_SIZE: int = 0  # unlimited
    PARQUET_DEFAULT_PASS_LIMIT: int = 16 * 1024**3  # 16GiB

    def __init__(
        self,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        cloud_options: dict[str, Any] | None,
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: expr.NamedExpr | None,
        parquet_options: None = None,
    ):
        self.schema = schema
        self.typ = typ
        self.reader_options = reader_options
        self.cloud_options = cloud_options
        self.paths = paths
        self.with_columns = with_columns
        self.skip_rows = skip_rows
        self.n_rows = n_rows
        self.row_index = row_index
        self.include_file_paths = include_file_paths
        self.predicate = predicate
        self._non_child_args = (
            schema,
            typ,
            reader_options,
            paths,
            with_columns,
            skip_rows,
            n_rows,
            row_index,
            include_file_paths,
            predicate,
            parquet_options,
        )
        self.children = ()
        self.parquet_options = parquet_options
        if self.typ not in ("csv", "parquet", "ndjson"):  # pragma: no cover
            # This line is unhittable ATM since IPC/Anonymous scan raise
            # on the polars side
            raise NotImplementedError(f"Unhandled scan type: {self.typ}")
        if self.typ == "ndjson" and (self.n_rows != -1 or self.skip_rows != 0):
            raise NotImplementedError("row limit in scan for json reader")
        if self.skip_rows < 0:
            # TODO: polars has this implemented for parquet,
            # maybe we can do this too?
            raise NotImplementedError("slice pushdown for negative slices")
        if (
            POLARS_VERSION_LT_128 and self.typ in {"csv"} and self.skip_rows != 0
        ):  # pragma: no cover
            # This comes from slice pushdown, but that
            # optimization doesn't happen right now
            raise NotImplementedError("skipping rows in CSV reader")
        if self.cloud_options is not None and any(
            self.cloud_options.get(k) is not None for k in ("aws", "azure", "gcp")
        ):
            raise NotImplementedError(
                "Read from cloud storage"
            )  # pragma: no cover; no test yet
        if any(str(p).startswith("https:/") for p in self.paths):
            raise NotImplementedError("Read from https")
        if self.typ == "csv":
            if self.reader_options["skip_rows_after_header"] != 0:
                raise NotImplementedError("Skipping rows after header in CSV reader")
            parse_options = self.reader_options["parse_options"]
            if (
                null_values := parse_options["null_values"]
            ) is not None and "Named" in null_values:
                raise NotImplementedError(
                    "Per column null value specification not supported for CSV reader"
                )
            if (
                comment := parse_options["comment_prefix"]
            ) is not None and "Multi" in comment:
                raise NotImplementedError(
                    "Multi-character comment prefix not supported for CSV reader"
                )
            if not self.reader_options["has_header"]:
                # TODO: To support reading headerless CSV files without requiring new
                # column names, we would need to do file introspection to infer the number
                # of columns so column projection works right.
                reader_schema = self.reader_options.get("schema")
                if not (
                    reader_schema
                    and isinstance(schema, dict)
                    and "fields" in reader_schema
                ):
                    raise NotImplementedError(
                        "Reading CSV without header requires user-provided column names via new_columns"
                    )
        elif self.typ == "ndjson":
            # TODO: consider handling the low memory option here
            # (maybe use chunked JSON reader)
            if self.reader_options["ignore_errors"]:
                raise NotImplementedError(
                    "ignore_errors is not supported in the JSON reader"
                )
            if include_file_paths is not None:
                # TODO: Need to populate num_rows_per_source in read_json in libcudf
                raise NotImplementedError("Including file paths in a json scan.")
        elif (
            self.typ == "parquet"
            and self.row_index is not None
            and self.with_columns is not None
            and len(self.with_columns) == 0
        ):
            raise NotImplementedError(
                "Reading only parquet metadata to produce row index."
            )

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The options dictionaries are serialised for hashing purposes
        as json strings.
        """
        schema_hash = tuple(self.schema.items())
        return (
            type(self),
            schema_hash,
            self.typ,
            json.dumps(self.reader_options),
            json.dumps(self.cloud_options),
            tuple(self.paths),
            tuple(self.with_columns) if self.with_columns is not None else None,
            self.skip_rows,
            self.n_rows,
            self.row_index,
            self.include_file_paths,
            self.predicate,
            self.parquet_options,
        )

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        typ: str,
        reader_options: dict[str, Any],
        paths: list[str],
        with_columns: list[str] | None,
        skip_rows: int,
        n_rows: int,
        row_index: tuple[str, int] | None,
        include_file_paths: str | None,
        predicate: expr.NamedExpr | None,
        parquet_options: None = None,
    ):
        """Evaluate and return a dataframe."""
        if typ == "csv":
            parse_options = reader_options.pop("parse_options")
            sep = chr(parse_options.pop("separator"))
            quote = chr(parse_options.pop("quote_char"))
            if quote != '"':
                raise NotImplementedError(f"{quote=}")

            eol = chr(parse_options.pop("eol_char"))
            if eol != "\n":
                raise NotImplementedError(f"{eol=} not supported (yet?)")

            schema_from_options = reader_options.pop("schema")
            if schema_from_options is not None:
                # Reader schema provides names
                column_names = list(schema_from_options["fields"].keys())
            else:
                # file provides column names, assume schema holds this
                column_names = list(schema.keys())

            has_header = reader_options.pop("has_header")
            if not has_header:
                # Need to do some file introspection to get the number
                # of columns so that column projection works right.
                raise NotImplementedError("Reading CSV without header")

            usecols = with_columns
            # TODO: support has_header=False
            # header = 0

            # polars defaults to no null recognition
            if parse_options.pop("null_values") is not None:
                raise NotImplementedError("null_values not supported yet")
            if parse_options.pop("comment_prefix") is not None:
                raise NotImplementedError(
                    f"{parse_options['comment_prefix']=} not supported (yet?)"
                )

            decimal = "," if parse_options.pop("decimal_comma") else "."
            if decimal != ".":
                raise NotImplementedError(f"{decimal=} not supported (yet?)")
            # polars skips blank lines at the beginning of the file
            if n_rows != -1:
                raise NotImplementedError(f"{n_rows=}")

            skiprows = reader_options.pop("skip_rows")
            if not POLARS_VERSION_LT_128:
                skiprows += skip_rows  # pragma: no cover

            if parse_options.pop("encoding", "Utf8") != "Utf8":
                raise NotImplementedError(
                    f"{parse_options.pop('encoding')=} not supported."
                )
            if parse_options.pop("missing_is_null", True) is not True:
                raise NotImplementedError(
                    f"{parse_options.pop('missing_is_null')=} not supported."
                )
            if parse_options.pop("truncate_ragged_lines", False) is not False:
                raise NotImplementedError(
                    f"{parse_options.pop('truncate_ragged_lines')=} not supported."
                )
            if parse_options.pop("try_parse_dates", False) is not False:
                raise NotImplementedError(
                    f"{parse_options.pop('try_parse_dates')=} not supported."
                )

            if column_names is None:
                raise NotImplementedError(f"Column names is required? {column_names=}")
            if usecols is not None:
                column_names = usecols

            if skiprows != 0:
                # NOTE: If supporting skiprows, must also remove trailing empty lines
                # to match polars behavior.
                raise NotImplementedError(f"{skiprows=}")

            if len(parse_options) > 0:
                raise NotImplementedError(
                    f"Unhandled csv parsing options: {parse_options}"
                )

            table = csv.csv_read(
                paths,
                delimiter=sep,
                # header=header,  TODO: Should be fine to ignore?
                usecols=column_names,
                na_filter=True,
                # keep_default_na=False, TODO: Fixme
                dtypes=[schema[n] for n in column_names],
            )
            df = DataFrame.from_table(table)
        elif typ == "parquet":
            # TODO: We should support pushing some predicates into the parquet reader

            if n_rows != -1:
                raise NotImplementedError(
                    f"{n_rows=} only full read supported right now"
                )
            if skip_rows != 0:
                raise NotImplementedError(
                    f"{skip_rows=} only full read supported right now"
                )

            table = parquet.parquet_read(paths, columns=with_columns)
            df = DataFrame.from_table(table)
        else:
            raise NotImplementedError(
                f"Unhandled scan type: {typ}"
            )  # pragma: no cover; post init trips first

        if row_index is not None:
            raise NotImplementedError("Row index not implemented, please avoid it.")

        assert all(df.table[n].type() == schema[n] for n in df.column_names)
        if predicate is None:
            return df
        else:
            (mask,) = broadcast(predicate.evaluate(df), target_length=df.num_rows)
            return df.filter(mask)


class Cache(IR):
    """
    Return a cached plan node.

    Used for CSE at the plan level.
    """

    __slots__ = ("key",)
    _non_child = ("schema", "key")
    key: int
    """The cache key."""

    def __init__(self, schema: Schema, key: int, value: IR):
        self.schema = schema
        self.key = key
        self.children = (value,)
        self._non_child_args = (key,)

    @classmethod
    def do_evaluate(
        cls, key: int, df: DataFrame
    ) -> DataFrame:  # pragma: no cover; basic evaluation never calls this
        """Evaluate and return a dataframe."""
        # Our value has already been computed for us, so let's just
        # return it.
        return df

    def evaluate(
        self, *, cache: MutableMapping[int, DataFrame], timer: Timer | None
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # We must override the recursion scheme because we don't want
        # to recurse if we're in the cache.
        try:
            return cache[self.key]
        except KeyError:
            (value,) = self.children
            return cache.setdefault(self.key, value.evaluate(cache=cache, timer=timer))


class DataFrameScan(IR):
    """
    Input from an existing polars DataFrame.

    This typically arises from ``q.collect().lazy()``
    """

    __slots__ = ("_id_for_hash", "df", "projection")
    _non_child = ("schema", "df", "projection")
    df: Any
    """Polars internal PyDataFrame object."""
    projection: tuple[str, ...] | None
    """List of columns to project out."""

    def __init__(
        self,
        schema: Schema,
        df: Any,
        projection: Sequence[str] | None,
    ):
        self.schema = schema
        self.df = df
        self.projection = tuple(projection) if projection is not None else None
        self._non_child_args = (
            schema,
            pl.DataFrame._from_pydf(df),
            self.projection,
        )
        self.children = ()
        self._id_for_hash = random.randint(0, 2**64 - 1)

    def get_hashable(self) -> Hashable:
        """
        Hashable representation of the node.

        The (heavy) dataframe object is not hashed. No two instances of
        ``DataFrameScan`` will have the same hash, even if they have the
        same schema, projection, and config options, and data.
        """
        schema_hash = tuple(self.schema.items())
        return (
            type(self),
            schema_hash,
            self._id_for_hash,
            self.projection,
        )

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        df: Any,
        projection: tuple[str, ...] | None,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if projection is not None:
            df = df.select(projection)
        df = DataFrame.from_polars(df)
        assert all(
            c.obj.type() == dtype
            for c, dtype in zip(df.columns, schema.values(), strict=True)
        )
        return df


class Select(IR):
    """Produce a new dataframe selecting given expressions from an input."""

    __slots__ = ("exprs", "should_broadcast")
    _non_child = ("schema", "exprs", "should_broadcast")
    exprs: tuple[expr.NamedExpr, ...]
    """List of expressions to evaluate to form the new dataframe."""
    should_broadcast: bool
    """Should columns be broadcast?"""

    def __init__(
        self,
        schema: Schema,
        exprs: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.exprs = tuple(exprs)
        self.should_broadcast = should_broadcast
        self.children = (df,)
        self._non_child_args = (self.exprs, should_broadcast)

    @classmethod
    def do_evaluate(
        cls,
        exprs: tuple[expr.NamedExpr, ...],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # Handle any broadcasting
        columns = [e.evaluate(df) for e in exprs]
        if should_broadcast:
            columns = broadcast(*columns)
        return DataFrame(columns)


class Reduce(IR):
    """
    Produce a new dataframe selecting given expressions from an input.

    This is a special case of :class:`Select` where all outputs are a single row.
    """

    __slots__ = ("exprs",)
    _non_child = ("schema", "exprs")
    exprs: tuple[expr.NamedExpr, ...]
    """List of expressions to evaluate to form the new dataframe."""

    def __init__(
        self, schema: Schema, exprs: Sequence[expr.NamedExpr], df: IR
    ):  # pragma: no cover; polars doesn't emit this node yet
        self.schema = schema
        self.exprs = tuple(exprs)
        self.children = (df,)
        self._non_child_args = (self.exprs,)

    @classmethod
    def do_evaluate(
        cls,
        exprs: tuple[expr.NamedExpr, ...],
        df: DataFrame,
    ) -> DataFrame:  # pragma: no cover; not exposed by polars yet
        """Evaluate and return a dataframe."""
        raise NotImplementedError("Should implement this, but need to change elsewhere")
        columns = broadcast(*(e.evaluate(df) for e in exprs))
        assert all(column.size == 1 for column in columns)
        return DataFrame(columns)


class Union(IR):
    """Concatenate dataframes vertically."""

    __slots__ = ("zlice",)
    _non_child = ("schema", "zlice")
    zlice: Zlice | None
    """Optional slice to apply to the result."""

    def __init__(self, schema: Schema, zlice: Zlice | None, *children: IR):
        self.schema = schema
        self.zlice = zlice
        self._non_child_args = (zlice,)
        self.children = children
        schema = self.children[0].schema

    @classmethod
    def do_evaluate(cls, zlice: Zlice | None, *dfs: DataFrame) -> DataFrame:
        """Evaluate and return a dataframe."""
        # TODO: only evaluate what we need if we have a slice?
        columns = []
        for name in dfs[0].column_names:
            new_column = copying.concatenate([df.table.get_column(name) for df in dfs])
            columns.append(Column(new_column, name=name))
        return DataFrame(columns).slice(zlice)


class HStack(IR):
    """Add new columns to a dataframe."""

    __slots__ = ("columns", "should_broadcast")
    _non_child = ("schema", "columns", "should_broadcast")
    should_broadcast: bool
    """Should the resulting evaluated columns be broadcast to the same length."""

    def __init__(
        self,
        schema: Schema,
        columns: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.columns = tuple(columns)
        self.should_broadcast = should_broadcast
        self._non_child_args = (self.columns, self.should_broadcast)
        self.children = (df,)

    @classmethod
    def do_evaluate(
        cls,
        exprs: Sequence[expr.NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # NOTE: This is fully unmodified from original implementation
        columns = [c.evaluate(df) for c in exprs]
        if should_broadcast:
            columns = broadcast(
                *columns, target_length=df.num_rows if df.num_columns != 0 else None
            )
        else:
            # Polars ensures this is true, but let's make sure nothing
            # went wrong. In this case, the parent node is a
            # guaranteed to be a Select which will take care of making
            # sure that everything is the same length. The result
            # table that might have mismatching column lengths will
            # never be turned into a pylibcudf Table with all columns
            # by the Select, which is why this is safe.
            assert all(e.name.startswith("__POLARS_CSER_0x") for e in exprs)
        return df.with_columns(columns)


class GroupBy(IR):
    """Perform a groupby."""

    __slots__ = (
        "agg_requests",
        "keys",
        "maintain_order",
        "zlice",
    )
    _non_child = (
        "schema",
        "keys",
        "agg_requests",
        "maintain_order",
        "zlice",
    )
    keys: tuple[expr.NamedExpr, ...]
    """Grouping keys."""
    agg_requests: tuple[expr.NamedExpr, ...]
    """Aggregation expressions."""
    maintain_order: bool
    """Preserve order in groupby."""
    zlice: Zlice | None
    """Optional slice to apply after grouping."""

    def __init__(
        self,
        schema: Schema,
        keys: Sequence[expr.NamedExpr],
        agg_requests: Sequence[expr.NamedExpr],
        maintain_order: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: IR,
    ):
        self.schema = schema
        self.keys = tuple(keys)
        for request in agg_requests:
            e = request.value
            if isinstance(e, expr.UnaryFunction) and e.name == "value_counts":
                raise NotImplementedError("value_counts is not supported in groupby")
            if any(
                isinstance(child, expr.UnaryFunction) and child.name == "value_counts"
                for child in e.children
            ):
                raise NotImplementedError("value_counts is not supported in groupby")
        self.agg_requests = tuple(agg_requests)
        self.maintain_order = maintain_order
        self.zlice = zlice
        self.children = (df,)

        self._non_child_args = (
            schema,
            self.keys,
            self.agg_requests,
            maintain_order,
            self.zlice,
        )

    @classmethod
    def do_evaluate(
        cls,
        schema: Schema,
        keys_in: Sequence[expr.NamedExpr],
        agg_requests: Sequence[expr.NamedExpr],
        maintain_order: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        keys = broadcast(*(k.evaluate(df) for k in keys_in), target_length=df.num_rows)

        if maintain_order:
            raise NotImplementedError(
                "maintain_order is not supported in groupby aggregations"
            )

        columns = {col.evaluate(df): col.name for i, col in enumerate(keys_in)}
        key_names = [n for n in columns.values()]

        requests = []
        for request in agg_requests:
            name = request.name
            value = request.value
            if isinstance(value, expr.Len):
                # A count aggregation, we need a column so use a key column
                col = keys[0]
            elif isinstance(value, expr.Agg):
                if value.name == "quantile":
                    raise NotImplementedError(
                        "quantile is not supported in groupby aggregations"
                    )
                (child,) = value.children
                col = child.evaluate(df)
            else:
                # Anything else, we pre-evaluate
                col = value.evaluate(df)

            if value.agg_request == "count_all":
                col_name = None
            else:
                col_name = columns.get(col, None)
                if col_name is None:
                    # NOTE(seberg): May need a unique name here eventually
                    columns[col] = (col_name := name)

            requests.append((col_name, value.agg_request, name))

        tbl = LogicalTable([c.obj for c in columns.keys()], columns.values())
        res_tbl = groupby_aggregation.groupby_aggregation(tbl, key_names, requests)
        # Handle order preservation of groups
        return DataFrame.from_table(res_tbl).slice(zlice)


class Join(IR):
    """A join of two dataframes."""

    __slots__ = ("config_options", "left_on", "options", "right_on")
    _non_child = ("schema", "left_on", "right_on", "options", "config_options")
    left_on: tuple[expr.NamedExpr, ...]
    """List of expressions used as keys in the left frame."""
    right_on: tuple[expr.NamedExpr, ...]
    """List of expressions used as keys in the right frame."""
    options: tuple[
        Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
        bool,
        Zlice | None,
        str,
        bool,
        Literal["none", "left", "right", "left_right", "right_left"],
    ]
    """
    tuple of options:
    - how: join type
    - nulls_equal: do nulls compare equal?
    - slice: optional slice to perform after joining.
    - suffix: string suffix for right columns if names match
    - coalesce: should key columns be coalesced (only makes sense for outer joins)
    - maintain_order: which DataFrame row order to preserve, if any
    """
    config_options: ConfigOptions
    """GPU-specific configuration options"""

    def __init__(
        self,
        schema: Schema,
        left_on: Sequence[expr.NamedExpr],
        right_on: Sequence[expr.NamedExpr],
        options: Any,
        config_options: ConfigOptions,
        left: IR,
        right: IR,
    ):
        self.schema = schema
        self.left_on = tuple(left_on)
        self.right_on = tuple(right_on)
        self.options = options
        self.config_options = config_options
        self.children = (left, right)
        self._non_child_args = (self.left_on, self.right_on, self.options)
        # TODO: Implement maintain_order
        if options[5] != "none":
            raise NotImplementedError("maintain_order not supported")

    @classmethod
    def do_evaluate(
        cls,
        left_on_exprs: Sequence[expr.NamedExpr],
        right_on_exprs: Sequence[expr.NamedExpr],
        options: tuple[
            Literal["Inner", "Left", "Right", "Full", "Semi", "Anti", "Cross"],
            bool,
            Zlice | None,
            str,
            bool,
            Literal["none", "left", "right", "left_right", "right_left"],
        ],
        left: DataFrame,
        right: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        how, nulls_equal, zlice, suffix, coalesce, maintain_order = options
        # Note that Left joins in older polars maintained order, we ignore this.
        assert maintain_order == "none"

        if how == "Cross":
            # Separate implementation, since cross_join returns the
            # result, not the gather maps
            raise NotImplementedError("Cross join")

        # TODO: Not clear that all are named expressions here, need to bloat table
        # to support this.
        if not all(isinstance(left, expr.NamedExpr) for left in left_on_exprs):
            raise NotImplementedError("Left on must be named expressions currently")
        if not all(isinstance(right, expr.NamedExpr) for right in right_on_exprs):
            raise NotImplementedError("Right on must be named expressions currently")

        left_on = [left.name for left in left_on_exprs]
        right_on = [right.name for right in right_on_exprs]

        if how == "Inner":
            join_type = join.JoinType.INNER
        elif how == "Left":
            join_type = join.JoinType.LEFT
        elif how == "Full":
            join_type = join.JoinType.FULL
        else:
            raise NotImplementedError(f"{how=}")

        left_tbl, right_tbl = left.table, right.table
        lhs_out_columns = left_tbl.get_column_names()
        rhs_out_columns = right_tbl.get_column_names()

        # Rename right columns to avoid any clashes:
        right_on = [r + suffix if r in lhs_out_columns else r for r in right_on]
        rhs_out_columns = [
            r + suffix if r in lhs_out_columns else r for r in rhs_out_columns
        ]
        orig_names = right_tbl.get_column_names()
        right_tbl = LogicalTable([right_tbl[n] for n in orig_names], rhs_out_columns)

        # Now, drop the join column if coalesce is given
        assert coalesce is not None
        if coalesce:
            for name in right_on:
                rhs_out_columns.remove(name)

        df = join.join(
            left_tbl,
            right_tbl,
            join_type=join_type,
            lhs_keys=left_on,
            rhs_keys=right_on,
            lhs_out_columns=lhs_out_columns,
            rhs_out_columns=rhs_out_columns,
            nulls_equal=nulls_equal,
        )

        return DataFrame.from_table(df).slice(zlice)


class Sort(IR):
    """Sort a dataframe."""

    __slots__ = ("by", "nulls_at_end", "sort_ascending", "stable", "zlice")
    _non_child = ("schema", "by", "sort_ascending", "nulls_at_end", "stable", "zlice")
    by: tuple[expr.NamedExpr, ...]
    """Sort keys."""
    sort_ascending: tuple[bool, ...]
    """Sort order for each sort key."""
    nulls_at_end: bool
    """Null sorting location for each sort key."""
    stable: bool
    """Should the sort be stable?"""
    zlice: Zlice | None
    """Optional slice to apply to the result."""

    def __init__(
        self,
        schema: Schema,
        by: Sequence[expr.NamedExpr],
        sort_ascending: Sequence[bool],
        nulls_at_end: bool,
        stable: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: IR,
    ):
        self.schema = schema
        self.by = tuple(by)
        self.sort_ascending = tuple(sort_ascending)
        self.nulls_at_end = nulls_at_end
        self.stable = stable
        self.zlice = zlice
        self._non_child_args = (
            self.by,
            self.sort_ascending,
            self.nulls_at_end,
            self.stable,
            self.zlice,
        )
        self.children = (df,)

    @classmethod
    def do_evaluate(
        cls,
        by: Sequence[expr.NamedExpr],
        sort_ascending: Sequence[bool],
        nulls_at_end: bool,
        stable: bool,  # noqa: FBT001
        zlice: Zlice | None,
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        if not all(isinstance(key, expr.NamedExpr) for key in by):
            raise NotImplementedError("Sort keys must be named expressions currently")
        by_names = [key.name for key in by]

        if zlice is None:
            limit = None
        elif zlice[0] < 0:
            # Assume the best slice is slicing from the end
            limit = zlice[0]
        else:
            if zlice[1] is None:
                limit = None
            else:
                limit = zlice[0] + zlice[1]

        table = sort.sort(
            df.table,
            by_names,
            sort_ascending=list(sort_ascending),
            nulls_at_end=nulls_at_end,
            stable=stable,
            limit=limit,
        )

        # TODO: should implement simple zlices into sorting probably.
        # (because head/tail can be applied locally skipping a lot of work).
        return DataFrame.from_table(table).slice(zlice)


class Slice(IR):
    """Slice a dataframe."""

    __slots__ = ("length", "offset")
    _non_child = ("schema", "offset", "length")
    offset: int
    """Start of the slice."""
    length: int
    """Length of the slice."""

    def __init__(self, schema: Schema, offset: int, length: int, df: IR):
        self.schema = schema
        self.offset = offset
        self.length = length
        self._non_child_args = (offset, length)
        self.children = (df,)

    @classmethod
    def do_evaluate(cls, offset: int, length: int, df: DataFrame) -> DataFrame:
        """Evaluate and return a dataframe."""
        return df.slice((offset, length))


class Filter(IR):
    """Filter a dataframe with a boolean mask."""

    __slots__ = ("mask",)
    _non_child = ("schema", "mask")
    mask: expr.NamedExpr
    """Expression to produce the filter mask."""

    def __init__(self, schema: Schema, mask: expr.NamedExpr, df: IR):
        self.schema = schema
        self.mask = mask
        self._non_child_args = (mask,)
        self.children = (df,)

    @classmethod
    def do_evaluate(cls, mask_expr: expr.NamedExpr, df: DataFrame) -> DataFrame:
        """Evaluate and return a dataframe."""
        (mask,) = broadcast(mask_expr.evaluate(df), target_length=df.num_rows)
        return df.filter(mask)


class Projection(IR):
    """Select a subset of columns from a dataframe."""

    __slots__ = ()
    _non_child = ("schema",)

    def __init__(self, schema: Schema, df: IR):
        self.schema = schema
        self._non_child_args = (schema,)
        self.children = (df,)

    @classmethod
    def do_evaluate(cls, schema: Schema, df: DataFrame) -> DataFrame:
        """Evaluate and return a dataframe."""
        # This can reorder things.
        columns = broadcast(
            *(df.column_map[name] for name in schema), target_length=df.num_rows
        )
        return DataFrame(columns)
