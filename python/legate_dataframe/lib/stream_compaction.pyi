# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from legate_dataframe import LogicalColumn, LogicalTable

__all__ = ["apply_boolean_mask"]

def apply_boolean_mask(
    tbl: LogicalTable, boolean_mask: LogicalColumn
) -> LogicalTable: ...
