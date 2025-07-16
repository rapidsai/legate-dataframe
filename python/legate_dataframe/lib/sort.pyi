# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.types import NullOrder, Order

from legate_dataframe.lib.core.table import LogicalTable

__all__ = ["NullOrder", "Order", "sort"]

def sort(
    tbl: LogicalTable,
    keys: list[str],
    *,
    sort_ascending: list[bool] | None,
    nulls_at_end: bool = True,
    stable: bool,
) -> LogicalTable: ...
