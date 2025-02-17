# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal
import warnings

import cudf
import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends import BaseBackend, NoUrl
from ibis.common.dispatch import lazy_singledispatch

import legate_dataframe.lib.csv
import legate_dataframe.lib.parquet
from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.experimental_ibis.executor import execute
from legate_dataframe.experimental_ibis.rewrites import rewrite_join
# from ibis.backends.pandas.rewrites import rewrite_join
from legate_dataframe.experimental_ibis.schema import (
    get_names_dtypes_from_schema,
    infer_schema_from_logical_table,
)
from legate_dataframe.experimental_ibis.utils import _gen_name

__all__ = ["LegateBackend"]

class LegateBackend(BaseBackend, NoUrl):
    name = "legate_dataframe"
    dialect = None

    class Options(ibis.config.Config):
        enable_trace: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = dict()

    def do_connect(
        self,
        tables: Mapping[str, LogicalTable] | None = None,
    ) -> None:
        """Construct a client from a mapping of `legate_dataframe.LogicalTable`
        or coercible tables (cudf, arrow, pandas).

        Parameters
        ----------
        tables
            Optional mapping with table names to table objects.
        """
        self._tables.clear()

        if tables is None:
            return
        for name, table in tables.items():
            self.create_table(name, table)

    def disconnect(self) -> None:
        self._tables.clear()

    @property
    def version(self) -> str:
        return legate_dataframe.__version__

    def read_csv(
        self,
        source: str | pathlib.Path,
        table_name: str | None = None,
        *,
        schema: sch.Schema,
        delimiter: str = ",",
    ):
        """read-csv version for legate.  Requires the schema to be passed
        currently.
        """
        # TODO(seberg): read_csv should return a (lazy) expression.
        if table_name is None:
            table_name = _gen_name("read_csv", str(source))

        usecols, dtypes = get_names_dtypes_from_schema(schema)

        # TODO: May need to adjust "source" for some file schemas.
        table = legate_dataframe.lib.csv.csv_read(
            source, delimiter=delimiter, usecols=usecols, dtypes=dtypes
        )

        # We already have the schema, just insert:
        self._tables[table_name] = (table, schema)
        return self.table(table_name)

    def read_parquet(
        self,
        source: str | pathlib.Path,
        table_name: str | None = None,
        *,
        columns = None
    ):
        # TODO(seberg): read_parquet should return a (lazy) expression and
        # eventually auto project `columns=` so it is probably just not needed.
        # (but even if lazy, it needs to eagerly find the schema).
        if table_name is None:
            table_name = _gen_name("read_parquet", str(source))

        table = legate_dataframe.lib.parquet.parquet_read(source, columns=columns)
        return self.create_table(table_name, table)

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self._tables.keys()), like)

    def table(self, name: str, schema: sch.Schema | None = None):
        inferred_schema = self.get_schema(name)
        overridden_schema = {**inferred_schema, **(schema or {})}
        return ops.DatabaseTable(name, overridden_schema, self).to_expr()

    def get_schema(self, table_name, *, database=None):
        _, schema = self._tables[table_name]

        return schema

    def compile(self, expr, *args, **kwargs):
        # We don't have a different intermediate representation, so do nothing
        return expr

    def create_table(
        self,
        name: str,
        obj=None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table."""
        if temp is not None and not temp:
            com.IbisError(
                "Passing `temp=False` to the Legate backend's `create_table()` method "
                "is not supported: all tables are in memory and temporary."
            )

        if database is not None:
            com.IbisError(
                "Passing `database` to the Legate backend's `create_table()` method "
                "is not supported: Legate cannot set a database."
            )

        if name in self._tables and not overwrite:
            raise com.IbisError(f"Cannot overwrite existing table `{name}`")

        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")

        if schema is not None:
            schema = ibis.schema(schema)

        if obj is not None:
            table = self._convert_object(obj)
            if schema is None:
                schema = infer_schema_from_logical_table(table)
            else:
                actual_schema = infer_schema_from_logical_table(table)
                if schema != actual_schema:
                    raise com.IbisError(
                        f"Schema {schema} and inferred one {actual_schema} do not match."
                    )
        else:
            raise NotImplementedError("Cannot yet create empty table from schema.")

        self._tables[name] = table, schema

        return self.table(name)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        return self.create_table(
            name, obj=obj, temp=None, database=database, overwrite=overwrite
        )

    def drop_table(self, name: str, *, force: bool = False) -> None:
        if self._tables.pop(name, None) is None and not force:
            raise com.IbisError(f"Table {name} does not exist") from None

    def drop_view(self, name: str, *, force: bool = False) -> None:
        self.drop_table(name, force=force)

    def _convert_object(self, obj: Any) -> Any:
        return _convert_object(obj, self)

    @classmethod
    @lru_cache
    def _get_operations(cls):
        return tuple(op for op in _execute_node.registry if issubclass(op, ops.Value))

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        op_classes = cls._get_operations()
        return operation in op_classes or issubclass(operation, op_classes)

    def _to_ldf_table(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | Literal["default"] | None = None,
        **kwargs: Any,
    ) -> LogicalTable:
        """Execute after ensuring that the result is a logical table."""
        # TODO(seberg): we do not support (and ignore) the limit parameter for now.
        table_expr = expr.as_table()

        node = table_expr.op()
        node = node.replace(rewrite_join, context={"params": params, "backend": self})

        res = execute(node, backend=self, params=params, cache={})

        actual_schema = infer_schema_from_logical_table(res)
        if actual_schema != node.schema:
            # TODO(seberg): Happens with decimals quickly, so for now
            # just give a warning
            warnings.warn(f"Result schema {actual_schema} and expected one {node.schema} do not match.",
            UserWarning,
        )

        return res

    def to_legate(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | Literal["default"] | None = None,
        **kwargs: Any,
    ) -> LogicalTable | LogicalColumn:
        """Execute the expression and return a legate table, column, or scalar."""
        df = self._to_ldf_table(expr, params=params, limit=limit, **kwargs)

        if isinstance(expr, ir.Table):
            return df
        elif isinstance(expr, ir.Column):
            return df.get_column(0)
        elif isinstance(expr, ir.Scalar):
            raise NotImplementedError("Legate doesn't support scalar returns yet")
            # The following should work, but may want use `pylibcudf`` scalars
            # return df.to_cudf().iloc[0, 0]
        else:
            raise RuntimeError(f"Invalid result type expr {expr}.")

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | Literal["default"] | None = None,
        **kwargs: Any,
    ):
        # Execute is a misnomer for converting `to_pandas`.
        df = self._to_ldf_table(expr, params=params, limit=limit, **kwargs)
        pandas_df = df.to_cudf().to_pandas()

        if isinstance(expr, ir.Table):
            return pandas_df
        elif isinstance(expr, ir.Column):
            return pandas_df.iloc[:, 0]
        elif isinstance(expr, ir.Scalar):
            # The following should work, but may want use `pylibcudf`` scalars
            # return df.to_cudf().iloc[0, 0]
            raise NotImplementedError("Legate doesn't support scalar returns yet")
        else:
            raise RuntimeError(f"Invalid result type expr {expr}.")

    def _create_cached_table(self, name, expr):
        return self.create_table(name, expr.execute())

    def _drop_cached_table(self, name):
        del self._tables[name]

    def _finalize_memtable(self, name: str) -> None:
        """No-op, let Python handle clean up."""


@lazy_singledispatch
def _convert_object(obj: Any, _conn):
    raise com.BackendConversionError(
        f"Unable to convert {obj.__class__} object to backend type: {LogicalTable}"
    )


@_convert_object.register(LogicalTable)
def _logical_table(obj: Any, _conn):
    return obj


@_convert_object.register(ops.InMemoryTable)
def _table(obj, _conn):
    return _convert_object(obj.data.to_frame(), _conn)


@_convert_object.register("cudf.DataFrame")
def _cudf(obj, _conn):
    return LogicalTable.from_cudf(obj)


@_convert_object.register("pyarrow.Table")
def _pyarrow(obj, _conn):
    return LogicalTable.from_cudf(cudf.DataFrame.from_pyarrow(obj))


@_convert_object.register("pandas.DataFrame")
def _pandas(obj, _conn):
    return LogicalTable.from_cudf(cudf.DataFrame.from_pandas(obj))
