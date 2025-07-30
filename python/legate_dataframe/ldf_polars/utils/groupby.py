# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for grouped aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from legate_dataframe.ldf_polars.dsl import ir
from legate_dataframe.ldf_polars.utils.aggregations import apply_pre_evaluation
from legate_dataframe.ldf_polars.utils.naming import unique_names

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from legate_dataframe.ldf_polars.dsl import expr
    from legate_dataframe.ldf_polars.typing import Schema

__all__ = ["rewrite_groupby"]


def rewrite_groupby(
    node: Any,
    schema: Schema,
    keys: Sequence[expr.NamedExpr],
    aggs: Sequence[expr.NamedExpr],
    inp: ir.IR,
) -> ir.IR:
    """
    Rewrite a groupby plan node into something we can handle.

    Parameters
    ----------
    node
        The polars groupby plan node.
    schema
        Schema of the groupby plan node.
    keys
        Grouping keys.
    aggs
        Originally requested aggregations.
    inp
        Input plan node to the groupby.

    Returns
    -------
    New plan node representing the grouped aggregations.

    Raises
    ------
    NotImplementedError
        If any of the requested aggregations are unsupported.

    Notes
    -----
    Since libcudf can only perform grouped aggregations on columns
    (not arbitrary expressions), the approach is to split each
    aggregation into a pre-selection phase (evaluating expressions
    that live within an aggregation), the aggregation phase (now
    acting on columns only), and a post-selection phase (evaluating
    expressions of aggregated results).

    This does scheme does not permit nested aggregations, so those are
    unsupported.
    """
    # TODO: Should special case len(aggs) == 0 via stream compaction
    #        (although could potentially do that also later).

    aggs, group_schema, apply_post_evaluation = apply_pre_evaluation(
        schema, keys, aggs, unique_names(schema.keys())
    )
    # TODO: use Distinct when the partitioned executor supports it if
    # the requested aggregations are empty
    inp = ir.GroupBy(
        group_schema,
        keys,
        aggs,
        node.maintain_order,
        node.options.slice,
        inp,
    )
    return apply_post_evaluation(inp)
