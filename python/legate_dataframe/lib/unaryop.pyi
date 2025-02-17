# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.unary import unary_operator

from legate_dataframe.lib.core.column import LogicalColumn

__all__ = ["unary_operator", "unary_operation"]

def unary_operation(col: LogicalColumn, op: unary_operator) -> LogicalColumn: ...
