# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn

def unary_operation(col: LogicalColumn, op: str) -> LogicalColumn: ...
def cast(col: LogicalColumn, dtype: DTypeLike) -> LogicalColumn: ...
def round(col: LogicalColumn, digits: int, mode: str) -> LogicalColumn: ...
