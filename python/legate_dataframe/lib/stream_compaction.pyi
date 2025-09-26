# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from legate_dataframe import LogicalColumn, LogicalTable

__all__ = ["apply_boolean_mask"]

def apply_boolean_mask(
    tbl: LogicalTable, boolean_mask: LogicalColumn
) -> LogicalTable: ...
def distinct(
    tbl: LogicalTable,
    keys: Iterable[str] | None = None,
    *,
    high_cardinality: bool = False,
) -> LogicalTable: ...
