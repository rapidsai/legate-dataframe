# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from legate_dataframe.lib.core.table import LogicalTable

__all__ = ["sort"]

def sort(
    tbl: LogicalTable,
    keys: list[str],
    *,
    sort_ascending: list[bool] | None,
    nulls_at_end: bool = True,
    stable: bool,
) -> LogicalTable: ...
