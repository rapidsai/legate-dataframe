# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import defaultdict
from functools import singledispatch
from typing import List, Tuple

import cudf  # for scalar
import ibis.expr.operations as ops
from ibis.common.exceptions import OperationNotDefinedError, UnboundExpressionError
from ibis.common.graph import Graph
from ibis.common.patterns import InstanceOf

import legate_dataframe.lib.join as ldf_join
from legate_dataframe import LogicalTable
from legate_dataframe.experimental_ibis.schema import to_plc_type
from legate_dataframe.experimental_ibis.rewrites import RenameColumns, SingleJoin
from legate_dataframe.lib.binaryop import binary_operation, binary_operator
from legate_dataframe.lib.groupby_aggregation import (
    AggregationKind,
    groupby_aggregation,
)
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.lib.unaryop import unary_operation, unary_operator
from legate_dataframe.lib.sort import NullOrder, Order, sort as ldf_sort


UNCACHED = object()


def execute(expr, *, backend, params, cache):
    """Execute nodes recursively while caching intermediate results.

    This approach relies on non-blocking task submission in legate, so that
    cleaning up the ``cache`` only at the end of submitting all tasks is
    sufficient.
    If it is not, we could customize execution, although this may require
    re-writing e.g. join and aggregate since their children do not represent
    concrete intermediate results directly.
    """
    from legate.core import get_legate_runtime
    res = cache.get(expr, UNCACHED)
    if res is UNCACHED:
        print(f"    (Entering execution of {expr})")
        res = _execute(expr, backend=backend, params=params, cache=cache)
        print("    Finished (or finishing) executing", expr)
        get_legate_runtime().issue_execution_fence(block=True)
        print("           done")
        cache[expr] = res

    return res


@singledispatch
def _execute(expr, *, backend, params, cache):
    """Raw execution of a single node, use ``execute`` for a cached version.
    """
    raise NotImplementedError(
        f"Operation {expr!r} is not implemented for the legate backend"
    )


@_execute.register
def operation(op: ops.Node, **_):
    raise OperationNotDefinedError(f"No translation rule for {type(op)}")


@_execute.register
def literal(op: ops.Literal, **_):
    # TODO(seberg): The best may be to start using arrow scalars throughout.
    # (Since we use host scalars for task launches anyway.)
    dtype = op.dtype.to_pandas()
    return cudf.Scalar(op.value, dtype)


@_execute.register
def column(op: ops.Field, **kw):
    table = execute(op.rel, **kw)
    return table[op.name]


@_execute.register
def drop_columns(op: ops.DropColumns, **kw):
    # Note that in most cases we should query-optimize to push this as early
    # as possible (and e.g. integrate it into file reading).
    tbl = execute(op.parent, **kw)
    # values are the columns we need to keep already (could also execute them)
    names = op.values.keys()
    return LogicalTable([tbl[n] for n in names], names)

@_execute.register
def project(op: ops.Project, **kw):
    # Note: Will fail for scalars (and unnesting), which is fine, though.
    values = op.values.values()
    names = op.values.keys()

    columns = [execute(col, **kw) for col in values]
    return LogicalTable(columns, names)


@_execute.register
def filter(op: ops.Filter, **kw):
    tbl = execute(op.parent, **kw)
    if not op.predicates:
        return tbl

    # TODO(seberg): It would be nicer to re-write these to binary ops first.
    def _refine_mask(mask, other):
        other = execute(other, **kw)
        return binary_operation(mask, other, binary_operator.LOGICAL_AND, mask.dtype())

    mask = execute(op.predicates[0], **kw)
    mask = functools.reduce(_refine_mask, op.predicates[1:], mask)

    return apply_boolean_mask(tbl, mask)


_unaryops = {
    ops.Abs: unary_operator.ABS,
    ops.Acos: unary_operator.ARCCOS,
    ops.Asin: unary_operator.ARCSINH,
    ops.Atan: unary_operator.ARCTAN,
    ops.Ceil: unary_operator.CEIL,
    ops.Cos: unary_operator.COS,
    # ops.Cot:
    # ops.DayOfWeekIndex:
    ops.Exp: unary_operator.EXP,
    ops.Floor: unary_operator.FLOOR,
    # ops.IsInf:
    # ops.IsNan:
    # ops.IsNull:
    ops.Ln: unary_operator.LOG,
    # ops.Log10:
    # ops.Log2:
    # ops.Negate:
    ops.Not: unary_operator.NOT,
    # ops.NotNull:
    ops.Sin: unary_operator.SIN,
    ops.Sqrt: unary_operator.SQRT,
    ops.Tan: unary_operator.TAN,
    ops.BitwiseNot: unary_operator.BIT_INVERT,
}


@_execute.register
def unaryop(op: ops.Unary | ops.Comparison, **kw):
    # automatically pick the correct kernel based on the operand types
    typ = type(op)

    unaryop = _unaryops.get(typ)
    if unaryop is not None:
        col = execute(op.arg, **kw)
        return unary_operation(col, unaryop)

    raise OperationNotDefinedError(f"Operation {typ} not yet defined.")


_binops = {
    # math ops:
    ops.Atan2: binary_operator.ATAN2,
    ops.Add: binary_operator.ADD,
    ops.And: binary_operator.LOGICAL_AND,
    ops.Divide: binary_operator.TRUE_DIV,
    ops.FloorDivide: binary_operator.DIV,
    # ops.Modulus: (which rules to follow?)
    ops.Multiply: binary_operator.MUL,
    ops.Subtract: binary_operator.SUB,
    # logical/bitwise
    ops.And: binary_operator.LOGICAL_AND,
    ops.Or: binary_operator.LOGICAL_OR,
    # ops.Xor:
    ops.BitwiseOr: binary_operator.BITWISE_OR,
    ops.BitwiseXor: binary_operator.BITWISE_XOR,
    ops.BitwiseAnd: binary_operator.BITWISE_AND,
    # Comparisons:
    ops.Equals: binary_operator.EQUAL,
    ops.Less: binary_operator.LESS,
    ops.LessEqual: binary_operator.LESS_EQUAL,
    ops.Greater: binary_operator.GREATER,
    ops.GreaterEqual: binary_operator.GREATER_EQUAL,
    ops.NotEquals: binary_operator.NOT_EQUAL,
    ops.IdenticalTo: binary_operator.NULL_EQUALS,
    # Times (mostly probably identical as above):
    # ops.DateAdd: operator.add,
    # ops.DateSub: operator.sub,
    # ops.DateDiff: operator.sub,
    # ops.TimestampAdd: operator.add,
    # ops.TimestampSub: operator.sub,
    # ops.IntervalSubtract: operator.sub,
}


@_execute.register
def binaryop(op: ops.Binary | ops.Comparison, **kw):
    # automatically pick the correct kernel based on the operand types
    typ = type(op)
    dtype = to_plc_type(op.dtype)

    binop = _binops.get(typ)
    if binop is not None:
        lhs = execute(op.left, **kw)
        rhs = execute(op.right, **kw)
        return binary_operation(lhs, rhs, binop, dtype)

    raise OperationNotDefinedError(f"Operation {typ} not yet defined.")


reductions: dict[type, AggregationKind] = {
    ops.Min: AggregationKind.MIN,
    ops.Max: AggregationKind.MAX,
    ops.Sum: AggregationKind.SUM,
    ops.Mean: AggregationKind.MEAN,
    ops.Count: AggregationKind.COUNT_VALID,
    ops.CountStar: AggregationKind.COUNT_ALL,
    # ops.Mode:
    ops.Any: AggregationKind.ANY,
    ops.All: AggregationKind.ALL,
    ops.Median: AggregationKind.MEDIAN,
    # ops.ApproxMedian:
    # ops.BitAnd:
    # ops.BitOr:
    # ops.BitXor:
    # ops.Arbitrary: arbitrary,
    ops.CountDistinct: AggregationKind.NUNIQUE,
    # ops.ApproxCountDistinct:
}


@_execute.register
def reduction(op: ops.Reduction, **_):
    raise OperationNotDefinedError(
        "Reductions are not yet implemented (only groupby-aggs)"
    )


@_execute.register
def aggregation(op: ops.Aggregate, **kw):
    unique_cols = {expr: name for name, expr in op.groups.items()}
    keys = list(op.groups.keys())
    aggs: List[Tuple[str, AggregationKind, str]] = []
    reduce_col_counter = 0  # to generate unique names for reduce expressions

    if not op.groups:
        raise NotImplementedError("select is not implemented.")

    for res_name, v in op.metrics.items():
        if not isinstance(v, ops.Reduction):
            raise NotImplementedError("Groupby must be an aggregation.")
        if v.where is not None:
            # Note: could probably just as well push the error into execution.
            # Although, it may be plausible to push some where statements down.
            # (A group-by intermediate table could also make sense for this, though...)
            raise NotImplementedError("Where unsupported for groupby aggregations")

        col = v.arg

        if (red := reductions.get(type(v))) is None:
            raise NotImplementedError(f"reduction {type(v)} is not implemented.")

        if red == AggregationKind.COUNT_ALL:
            # Count need/has no column, but legate-df expects one currently.
            name = keys[0]
        else:
            # Use existing name or generate a new temporary one for the reduction:
            if (name := unique_cols.get(col)) is None:
                while (name := f"reduce_col_{reduce_col_counter}") in unique_cols.values():
                    reduce_col_counter += 1

                unique_cols[col] = name

        aggs.append((name, red, res_name))  # type: ignore

    columns = [execute(c, **kw) for c in unique_cols]
    tmp_table = LogicalTable(columns, unique_cols.values())

    return groupby_aggregation(tmp_table, keys, aggs)


_join_types = {
    "left": ldf_join.JoinType.LEFT,
    "inner": ldf_join.JoinType.INNER,
    "outer": ldf_join.JoinType.FULL,
}


@_execute.register
def unreference_joinref(op: ops.JoinReference, **kw):
    # We don't rewrite `JoinReferences` away, but if used it always refers
    # to the original table now.
    return execute(op.parent, **kw)


@_execute.register(RenameColumns)
def rename(op, **kw):
    tbl = execute(op.parent, **kw)
    columns = [tbl.get_column(i) for i in range(tbl.num_columns())]
    names = [op.mapping[name] for name in tbl.get_column_names()]
    return LogicalTable(columns, names)


def execute_table_with_columns(
    expr: ops.Relation, columns: list[ops.Value], prefix: str, kw,
) -> tuple[LogicalTable, list[str]]:
    """Execute a table and add columns if needed in the context.

    Helper to add columns prefixed by ``{prefix}{i}_``.  Normally i is just
    0, 1, etc. but it may skip one if the table already contains equivalent
    columns.  If the column already exists in the table, it simply returns
    the existing name.

    .. note::
        Currently assumes that added columns are definitely unique.

    Parameters
    ----------
    expr
        The table expression to execute.
    columns
        Additional columns to add
    prefix
        A prefix for the columns, the actual prefix is ``{prefix}{i}_``
        where ``i`` ensures uniqueness.

    Returns
    -------
    tbl
        The ``LogicalTable`` with all columns.
    tbl_column_names
        The column names originally in the table.
    colnames
        The names mapping to `columns` passed in.  These names may refer to
        existing names if the column is identical.
    """
    tbl = execute(expr, **kw)
    all_columns = [tbl.get_column(i) for i in range(tbl.num_columns())]
    all_colnames = tbl.get_column_names()
    prefix_counter = 0

    new_names = []
    for col in columns:
        if type(col) is ops.Field and col.rel == expr:
            new_names.append(col.name)
            continue

        all_columns.append(execute(col, **kw))
        # Just make extra sure the name is unique (imporant in some contexts)
        while (name := f"{prefix}{prefix_counter}_{col.name}") in all_colnames:
            prefix_counter += 1

        all_colnames.append(name)
        new_names.append(name)

    return LogicalTable(all_columns, all_colnames), tbl.get_column_names(), new_names


@_execute.register
def single_join(op: SingleJoin, **kw):
    left, left_cols, left_on = execute_table_with_columns(op.left, op.left_on, "left_on", kw)
    right, right_cols, right_on = execute_table_with_columns(op.right, op.right_on, "right_on", kw)

    if op.how == "inner":
        # For inner joins a single column predicate is the same as filtering
        # (since not joining the other table means omitting the result)
        # TODO(seberg): Works, but only good if the filter result is sparse.
        # Ibis polars/pandas does this by adding an all `true` column to the other one!
        # Need to add a helper for this.
        if op.left_filter is not None:
            left = apply_boolean_mask(left, op.left_filter)
        if op.right_filter is not None:
            right = apply_boolean_mask(right, op.right_on)
    elif op.left_filter is not None or op.right_filter is not None:
        raise NotImplementedError("Joins with single side predicates only work with inner")


    if op.how == "left":
        join_type = ldf_join.JoinType.LEFT
    elif op.how == "right":
        join_type = ldf_join.JoinType.LEFT
        left_table, right_table = right_table, left_table
        left_on, right_on = right_on, left_on
    elif op.how == "inner":
        join_type = ldf_join.JoinType.INNER
    else:
        raise NotImplementedError(f"join type {link.how} not implemented by legate.")

    res = ldf_join.join(
        left,
        right,
        lhs_keys=left_on,
        rhs_keys=right_on,
        join_type=join_type,
        lhs_out_columns=left_cols,
        rhs_out_columns=right_cols,
    )

    return res


@_execute.register
def sort(op: ops.Sort, **kw):
    tbl = execute(op.parent, **kw)
    if not op.keys:
        return tbl

    keys: list[str] = []
    column_order: list[Order] = []
    null_precedence: list[NullOrder] = []
    for key in op.keys:
        if type(key.expr) == ops.Field and key.expr.rel is op.parent:
            name = key.expr.name
        else:
            # TODO: need to add temporary columns (and remove them again later)
            raise NotImplementedError("Sort columns not part of the table need to be added")

        keys.append(name)
        column_order.append(Order.DESCENDING if key.descending else Order.ASCENDING)
        # libcudf uses value logic for `null_recedence` so translate (if one
        # order is swapped nulls go first otherwise they go last).
        if key.descending == key.nulls_first:
            null_precedence.append(NullOrder.AFTER)
        else:
            null_precedence.append(NullOrder.BEFORE)

    return ldf_sort(tbl, keys, column_order=column_order, null_precedence=null_precedence)


@_execute.register
def table(op: ops.DatabaseTable, **_):
    tables = op.source._tables
    name = op.name
    try:
        return tables[name][0]
    except KeyError:
        raise UnboundExpressionError(
            f"{name} is not a table in the {op.source.name!r} backend, you "
            "probably tried to execute an expression without a data source"
        )


@_execute.register(ops.InMemoryTable)
def in_memory_table(op, *, backend, **_):
    return backend._convert_object(op)
