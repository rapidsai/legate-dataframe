# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.types import NullOrder, Order

from legate_dataframe.lib.core.table import LogicalTable

__all__ = ["NullOrder", "Order", "sort"]

def sort(
    tbl: LogicalTable,
    keys: list[str],
    *,
    column_order: list[Order] | None,
    null_precedence: list[NullOrder] | None,
    stable: bool,
) -> LogicalTable: ...
