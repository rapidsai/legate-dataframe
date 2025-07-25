# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Translate polars IR representation to ours."""

from __future__ import annotations

import copy
import json
from contextlib import AbstractContextManager, nullcontext
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.polars as plrs
import pyarrow as pa
import pylibcudf as plc
from polars.polars import _expr_nodes as pl_expr
from polars.polars import _ir_nodes as pl_ir

from legate_dataframe.ldf_polars.dsl import expr, ir
from legate_dataframe.ldf_polars.typing import Schema
from legate_dataframe.ldf_polars.utils import config, dtypes, sorting
from legate_dataframe.ldf_polars.utils.versions import POLARS_VERSION_LT_131

if TYPE_CHECKING:
    from polars import GPUEngine

    from legate_dataframe.ldf_polars.typing import NodeTraverser

__all__ = ["Translator", "translate_named_expr"]


class Translator:
    """
    Translates polars-internal IR nodes and expressions to our representation.

    Parameters
    ----------
    visitor
        Polars NodeTraverser object
    engine
        GPU engine configuration.
    """

    def __init__(self, visitor: NodeTraverser, engine: GPUEngine):
        self.visitor = visitor
        self.config_options = config.ConfigOptions(copy.deepcopy(engine.config))
        self.errors: list[Exception] = []

    def translate_ir(self, *, n: int | None = None) -> ir.IR:
        """
        Translate a polars-internal IR node to our representation.

        Parameters
        ----------
        visitor
            Polars NodeTraverser object
        n
            Optional node to start traversing from, if not provided uses
            current polars-internal node.

        Returns
        -------
        Translated IR object

        Raises
        ------
        NotImplementedError
            If the version of Polars IR is unsupported.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorNode` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        ctx: AbstractContextManager[None] = (
            set_node(self.visitor, n) if n is not None else noop_context
        )
        # IR is versioned with major.minor, minor is bumped for backwards
        # compatible changes (e.g. adding new nodes), major is bumped for
        # incompatible changes (e.g. renaming nodes).
        if (version := self.visitor.version()) >= (8, 1):
            e = NotImplementedError(
                f"No support for polars IR {version=}"
            )  # pragma: no cover; no such version for now.
            self.errors.append(e)  # pragma: no cover
            raise e  # pragma: no cover

        with ctx:
            polars_schema = self.visitor.get_schema()
            try:
                schema = {k: dtypes.from_polars(v) for k, v in polars_schema.items()}
            except Exception as e:
                self.errors.append(NotImplementedError(str(e)))
                return ir.ErrorNode({}, str(e))
            try:
                node = self.visitor.view_current_node()
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            try:
                result = _translate_ir(node, self, schema)
            except Exception as e:
                self.errors.append(e)
                return ir.ErrorNode(schema, str(e))
            if any(
                isinstance(dtype, pl.Null)
                for dtype in pl.datatypes.unpack_dtypes(*polars_schema.values())
            ):
                error = NotImplementedError(
                    f"No GPU support for {result} with Null column dtype."
                )
                self.errors.append(error)
                return ir.ErrorNode(schema, str(error))

            return result

    def translate_expr(self, *, n: int, schema: Schema) -> expr.Expr:
        """
        Translate a polars-internal expression IR into our representation.

        Parameters
        ----------
        n
            Node to translate, an integer referencing a polars internal node.
        schema
            Schema of the IR node this expression uses as evaluation context.

        Returns
        -------
        Translated IR object.

        Notes
        -----
        Any expression nodes that cannot be translated are replaced by
        :class:`expr.ErrorExpr` nodes and collected in the the `errors` attribute.
        After translation is complete, this list of errors should be inspected
        to determine if the query is supported.
        """
        node = self.visitor.view_expression(n)
        dtype = dtypes.from_polars(self.visitor.get_dtype(n))
        try:
            return _translate_expr(node, self, dtype, schema)
        except Exception as e:
            self.errors.append(e)
            return expr.ErrorExpr(dtype, str(e))


class set_node(AbstractContextManager[None]):
    """
    Run a block with current node set in the visitor.

    Parameters
    ----------
    visitor
        The internal Rust visitor object
    n
        The node to set as the current root.

    Notes
    -----
    This is useful for translating expressions with a given node
    active, restoring the node when the block exits.
    """

    __slots__ = ("n", "visitor")
    visitor: NodeTraverser
    n: int

    def __init__(self, visitor: NodeTraverser, n: int) -> None:
        self.visitor = visitor
        self.n = n

    def __enter__(self) -> None:
        n = self.visitor.get_node()
        self.visitor.set_node(self.n)
        self.n = n

    def __exit__(self, *args: Any) -> None:
        self.visitor.set_node(self.n)


noop_context: nullcontext[None] = nullcontext()


@singledispatch
def _translate_ir(node: Any, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_ir.register
def _(node: pl_ir.PythonScan, translator: Translator, schema: Schema) -> ir.IR:
    scan_fn, with_columns, source_type, predicate, nrows = node.options
    options = (scan_fn, with_columns, source_type, nrows)
    predicate = (
        translate_named_expr(translator, n=predicate, schema=schema)
        if predicate is not None
        else None
    )
    return ir.PythonScan(schema, options, predicate)


@_translate_ir.register
def _(node: pl_ir.Scan, translator: Translator, schema: Schema) -> ir.IR:
    typ, *options = node.scan_type
    if typ == "ndjson":
        (reader_options,) = map(json.loads, options)
        cloud_options = None
    else:
        reader_options, cloud_options = map(json.loads, options)
    file_options = node.file_options
    with_columns = file_options.with_columns
    row_index = file_options.row_index
    include_file_paths = file_options.include_file_paths
    if not POLARS_VERSION_LT_131:
        deletion_files = file_options.deletion_files  # pragma: no cover
        if deletion_files:  # pragma: no cover
            raise NotImplementedError(
                "Iceberg format is not supported in legate-polars. "
                "Furthermore, row-level deletions are not supported."
            )  # pragma: no cover
    # config_options = translator.config_options
    # parquet_options = config_options.parquet_options

    pre_slice = file_options.n_rows
    if pre_slice is None:
        n_rows = -1
        skip_rows = 0
    else:
        skip_rows, n_rows = pre_slice

    return ir.Scan(
        schema,
        typ,
        reader_options,
        cloud_options,
        node.paths,
        with_columns,
        skip_rows,
        n_rows,
        row_index,
        include_file_paths,
        (
            translate_named_expr(translator, n=node.predicate, schema=schema)
            if node.predicate is not None
            else None
        ),
        # parquet_options,
    )


@_translate_ir.register
def _(node: pl_ir.Cache, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Cache(schema, node.id_, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(node: pl_ir.DataFrameScan, translator: Translator, schema: Schema) -> ir.IR:
    return ir.DataFrameScan(
        schema,
        node.df,
        node.projection,
    )


@_translate_ir.register
def _(node: pl_ir.Select, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.expr
        ]
    return ir.Select(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(node: pl_ir.GroupBy, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("GroupBy not supported")


@_translate_ir.register
def _(node: pl_ir.Join, translator: Translator, schema: Schema) -> ir.IR:
    # Join key dtypes are dependent on the schema of the left and
    # right inputs, so these must be translated with the relevant
    # input active.
    with set_node(translator.visitor, node.input_left):
        inp_left = translator.translate_ir(n=None)
        left_on = [
            translate_named_expr(translator, n=e, schema=inp_left.schema)
            for e in node.left_on
        ]
    with set_node(translator.visitor, node.input_right):
        inp_right = translator.translate_ir(n=None)
        right_on = [
            translate_named_expr(translator, n=e, schema=inp_right.schema)
            for e in node.right_on
        ]

    if (how := node.options[0]) in {
        "Inner",
        "Left",
        "Right",
        "Full",
        "Cross",
        "Semi",
        "Anti",
    }:
        return ir.Join(
            schema,
            left_on,
            right_on,
            node.options,
            translator.config_options,
            inp_left,
            inp_right,
        )
    else:
        raise NotImplementedError(f"ConditionalJoin with {how=} not supported")


@_translate_ir.register
def _(node: pl_ir.HStack, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.exprs
        ]
    return ir.HStack(schema, exprs, node.should_broadcast, inp)


@_translate_ir.register
def _(
    node: pl_ir.Reduce, translator: Translator, schema: Schema
) -> ir.IR:  # pragma: no cover; polars doesn't emit this node yet
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        exprs = [
            translate_named_expr(translator, n=e, schema=inp.schema) for e in node.expr
        ]
    return ir.Reduce(schema, exprs, inp)


@_translate_ir.register
def _(node: pl_ir.Distinct, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("Distinct not supported")


@_translate_ir.register
def _(node: pl_ir.Sort, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        by = [
            translate_named_expr(translator, n=e, schema=inp.schema)
            for e in node.by_column
        ]
    stable, nulls_last, descending = node.sort_options
    sort_ascending, nulls_at_end = sorting.sort_order(
        descending, nulls_last=nulls_last, num_keys=len(by)
    )
    return ir.Sort(schema, by, sort_ascending, nulls_at_end, stable, node.slice, inp)


@_translate_ir.register
def _(node: pl_ir.Slice, translator: Translator, schema: Schema) -> ir.IR:
    return ir.Slice(
        schema, node.offset, node.len, translator.translate_ir(n=node.input)
    )


@_translate_ir.register
def _(node: pl_ir.Filter, translator: Translator, schema: Schema) -> ir.IR:
    with set_node(translator.visitor, node.input):
        inp = translator.translate_ir(n=None)
        mask = translate_named_expr(translator, n=node.predicate, schema=inp.schema)
    return ir.Filter(schema, mask, inp)


@_translate_ir.register
def _(
    node: pl_ir.SimpleProjection,
    translator: Translator,
    schema: Schema,
) -> ir.IR:
    return ir.Projection(schema, translator.translate_ir(n=node.input))


@_translate_ir.register
def _(node: pl_ir.MergeSorted, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("MergeSorted not supported")


@_translate_ir.register
def _(node: pl_ir.MapFunction, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("MapFunction not supported")


@_translate_ir.register
def _(node: pl_ir.Union, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("Union not supported")


@_translate_ir.register
def _(node: pl_ir.HConcat, translator: Translator, schema: Schema) -> ir.IR:
    raise NotImplementedError("HConcat not supported")


def translate_named_expr(
    translator: Translator, *, n: pl_expr.PyExprIR, schema: Schema
) -> expr.NamedExpr:
    """
    Translate a polars-internal named expression IR object into our representation.

    Parameters
    ----------
    translator
        Translator object
    n
        Node to translate, a named expression node.
    schema
        Schema of the IR node this expression uses as evaluation context.

    Returns
    -------
    Translated IR object.

    Notes
    -----
    The datatype of the internal expression will be obtained from the
    visitor by calling ``get_dtype``, for this to work properly, the
    caller should arrange that the expression is translated with the
    node that it references "active" for the visitor (see :class:`set_node`).

    Raises
    ------
    NotImplementedError
        If any translation fails due to unsupported functionality.
    """
    return expr.NamedExpr(
        n.output_name, translator.translate_expr(n=n.node, schema=schema)
    )


@singledispatch
def _translate_expr(
    node: Any, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError(
        f"Translation for {type(node).__name__}"
    )  # pragma: no cover


@_translate_expr.register
def _(
    node: pl_expr.Function, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    name, *options = node.function_data
    options = tuple(options)
    if isinstance(name, pl_expr.StringFunction):
        raise NotImplementedError("StringFunction not supported")
    elif isinstance(name, pl_expr.BooleanFunction):
        if name == pl_expr.BooleanFunction.IsBetween:
            column, lo, hi = (
                translator.translate_expr(n=n, schema=schema) for n in node.input
            )
            (closed,) = options
            if closed == "none":
                lop, rop = "greater", "less"
            elif closed == "left":
                lop, rop = "greater_equal", "less"
            elif closed == "right":
                lop, rop = "greater", "less_equal"
            elif closed == "both":
                lop, rop = "greater_equal", "less_equal"
            else:
                raise NotImplementedError(f"IsBetween with {closed=} not supported")
            return expr.BinOp(
                dtype,
                "and_kleene",
                expr.BinOp(dtype, lop, column, lo),
                expr.BinOp(dtype, rop, column, hi),
            )
        raise NotImplementedError(
            f"BooleanFunction {name} not supported (only IsBetween is)"
        )
    elif isinstance(name, pl_expr.TemporalFunction):
        raise NotImplementedError("TemporalFunction not supported")

    elif isinstance(name, str):
        children = (translator.translate_expr(n=n, schema=schema) for n in node.input)
        if name == "log":
            (base,) = options
            (child,) = children
            return expr.BinOp(
                dtype,
                plc.binaryop.BinaryOperator.LOG_BASE,
                child,
                expr.Literal(dtype, pa.scalar(base, type=plc.interop.to_arrow(dtype))),
            )
        elif name == "pow":
            return expr.BinOp(dtype, plc.binaryop.BinaryOperator.POW, *children)
        elif name in "top_k":
            raise NotImplementedError("top_k not supported")

        return expr.UnaryFunction(dtype, name, options, *children)
    raise NotImplementedError(
        f"No handler for Expr function node with {name=}"
    )  # pragma: no cover; polars raises on the rust side for now


@_translate_expr.register
def _(node: pl_expr.Window, translator: Translator, dtype: plc.DataType) -> expr.Expr:
    # Note that this is for groupby operations.
    raise NotImplementedError("Window not supported")


@_translate_expr.register
def _(
    node: pl_expr.Literal, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    if isinstance(node.value, plrs.PySeries):
        return expr.LiteralColumn(dtype, pl.Series._from_pyseries(node.value))
    if dtype.id() == plc.TypeId.LIST:  # pragma: no cover
        # TODO: Remove once pylibcudf.Scalar supports lists
        return expr.LiteralColumn(dtype, pl.Series(node.value))
    return expr.Literal(dtype, node.value)


@_translate_expr.register
def _(
    node: pl_expr.Sort, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError("Sort not supported")


@_translate_expr.register
def _(
    node: pl_expr.SortBy, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError("SortBy not supported")


@_translate_expr.register
def _(
    node: pl_expr.Slice, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    offset = translator.translate_expr(n=node.offset, schema=schema)
    length = translator.translate_expr(n=node.length, schema=schema)
    assert isinstance(offset, expr.Literal)
    assert isinstance(length, expr.Literal)
    return expr.Slice(
        dtype,
        offset.value.as_py(),
        length.value.as_py(),
        translator.translate_expr(n=node.input, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Gather, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError("Gather not supported")


@_translate_expr.register
def _(
    node: pl_expr.Filter, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    return expr.Filter(
        dtype,
        translator.translate_expr(n=node.input, schema=schema),
        translator.translate_expr(n=node.by, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Cast, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    inner = translator.translate_expr(n=node.expr, schema=schema)
    # Push casts into literals so we can handle Cast(Literal(Null))
    if isinstance(inner, expr.Literal):
        return expr.Literal(dtype, inner.value.cast(plc.interop.to_arrow(dtype)))
    elif isinstance(inner, expr.Cast):
        # Translation of Len/Count-agg put in a cast, remove double
        # casts if we have one.
        (inner,) = inner.children
    return expr.Cast(dtype, inner)


@_translate_expr.register
def _(
    node: pl_expr.Column, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    return expr.Col(dtype, node.name)


@_translate_expr.register
def _(
    node: pl_expr.Agg, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    value = expr.Agg(
        dtype,
        node.name,
        node.options,
        *(translator.translate_expr(n=n, schema=schema) for n in node.arguments),
    )
    if value.name in ("count", "n_unique") and value.dtype.id() != plc.TypeId.INT32:
        return expr.Cast(value.dtype, value)
    return value


@_translate_expr.register
def _(
    node: pl_expr.Ternary, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    raise NotImplementedError("Ternary operations not supported")


@_translate_expr.register
def _(
    node: pl_expr.BinaryExpr,
    translator: Translator,
    dtype: plc.DataType,
    schema: Schema,
) -> expr.Expr:
    return expr.BinOp(
        dtype,
        expr.BinOp._MAPPING[node.op],
        translator.translate_expr(n=node.left, schema=schema),
        translator.translate_expr(n=node.right, schema=schema),
    )


@_translate_expr.register
def _(
    node: pl_expr.Len, translator: Translator, dtype: plc.DataType, schema: Schema
) -> expr.Expr:
    value = expr.Len(dtype)
    if dtype.id() != plc.TypeId.INT32:
        return expr.Cast(dtype, value)
    return value  # pragma: no cover; never reached since polars len has uint32 dtype
