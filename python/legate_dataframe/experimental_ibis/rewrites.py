# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import ibis.expr.operations as ops
from ibis.common.annotations import attribute
from ibis.common.collections import FrozenDict
from ibis.common.patterns import replace
import ibis.expr.schema as sch


# Note that these Join rewrites are heavily inspired from the pandas/polars
# ones in ibis:
# https://github.com/ibis-project/ibis/blob/main/ibis/backends/polars/rewrites.py
# (Also under Apache 2 license.)
class RenameColumns(ops.Relation):
    parent: ops.Relation
    mapping: FrozenDict[str, str]

    @classmethod
    def from_prefix(cls, parent, prefix):
        mapping = {k: f"{prefix}_{k}" for k in parent.schema}
        return cls(parent, mapping)

    @attribute
    def values(self):
        return FrozenDict(
            {to: ops.Field(self.parent, from_) for from_, to in self.mapping.items()}
        )

    @attribute
    def schema(self):
        return sch.Schema(
            {self.mapping[name]: dtype for name, dtype in self.parent.schema.items()}
        )


class SingleJoin(ops.Relation):
    # This is the same as the pandas join, but we do not use its renaming
    # design here (for now), but handle it during execution.
    left: ops.Relation
    right: ops.Relation
    left_on: tuple[ops.Value, ...]
    right_on: tuple[ops.Value, ...]
    left_filter: ops.Value | None
    right_filter: ops.Value | None
    how: str

    @attribute
    def values(self):
        return FrozenDict({**self.left.values, **self.right.values})

    @attribute
    def schema(self):
        return self.left.schema | self.right.schema


@replace(ops.JoinChain)
def rewrite_join(_, **kwargs):
    # TODO(seberg): This re-write drags along columns for longer than necessary.
    # However, I suspect this is a good way but we may want to re-write it to
    # drop columns early.
    left_table = RenameColumns.from_prefix(_.first, f"join{_.first.identifier}")

    for link in _.rest:
        right_table = RenameColumns.from_prefix(link.table, f"join{link.table.identifier}")

        # Same as substitution here in ibis.  Predicates refer to original cols
        # and we rewrite them here to refer to `Field(JoinReference, name)`
        # (or whichever table we now have).
        subs = {v: ops.Field(left_table, k) for k, v in left_table.values.items()}
        subs.update({v: ops.Field(right_table, k) for k, v in right_table.values.items()})
        predicates = [pred.replace(subs, filter=ops.Value) for pred in link.predicates]

        for pred in predicates:
            left_on = []
            right_on = []
            left_filter = []
            right_filter = []

            # If the predicate refers only to the left/right table we can
            # rewrite the predicates.  In the simplest case (inner join)
            # this can re-written as a filter.
            if left_table in pred.relations and len(pred.relations) == 1:
                left_filter.append(pred)
                continue
            elif right_table in pred.relations and len(pred.relations) == 1:
                right_filter.append(pred)
                continue

            # Otherwise, we only support equality predicates.
            if not isinstance(pred, ops.Equals):
                raise NotImplementedError(
                    "Join predicates must be equalities or refer to only one table.")

            # A normal join (columns should be equal), but we may need to swap
            if left_table in pred.left.relations and right_table in pred.right.relations:
                left_on.append(pred.left)
                right_on.append(pred.right)
            elif right_table in pred.left.relations and left_table in pred.right.relations:
                left_on.append(pred.right)
                right_on.append(pred.left)
            else:
                raise NotImplementedError(
                    "Join predicates must currently refer to the tables being joined.")

        # Add expressions that only work on one of the two tables (i.e. predicates
        # that must be true if joining should happen).
        # For an inner join this is a filter (but a filter may not be the fastest).
        if left_filter:
            left_filter_expr = functools.reduce(ops.And, left_filter)
        else:
            left_filter_expr = None

        if right_filter:
            right_filter_expr = functools.reduce(ops.And, right_filter)
        else:
            right_filter_expr = None

        # Currently, filter "asof" join during execution:
        left_table = SingleJoin(
            left_table, right_table, left_on, right_on,
            left_filter_expr, right_filter_expr, link.how)

    subs = {v: ops.Field(left_table, k) for k, v in left_table.values.items()}
    fields = {k: v.replace(subs, filter=ops.Value) for k, v in _.values.items()}
    return ops.Project(left_table, fields)
