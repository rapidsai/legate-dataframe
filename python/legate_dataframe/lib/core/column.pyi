# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import cudf.core.column
import legate.core
from cudf._typing import DtypeObj

class LogicalColumn:
    def __init__(self, obj: Any) -> None: ...
    @staticmethod
    def from_cudf(col: cudf.core.column.ColumnBase) -> LogicalColumn: ...
    @staticmethod
    def empty_like_logical_column(col: LogicalColumn) -> LogicalColumn: ...
    def num_rows(self) -> int: ...
    def dtype(self) -> DtypeObj: ...
    def get_logical_array(self) -> legate.core.LogicalArray: ...
    @property
    def __legate_data_interface__(self) -> dict: ...
    def to_cudf(self) -> cudf.core.column.ColumnBase: ...
    def repr(self, max_num_items: int) -> str: ...
    def __repr__(self) -> str: ...
    def add_as_next_task_input(self, task: legate.core.AutoTask) -> None: ...
    def add_as_next_task_output(self, task: legate.core.AutoTask) -> None: ...
