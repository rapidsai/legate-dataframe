# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, List

import cudf
from legate.core import StoreTarget

from legate_dataframe.lib.core.column import LogicalColumn

class LogicalTable:
    def __init__(
        self, columns: Iterable[LogicalColumn], column_names: Iterable[str]
    ) -> None: ...
    @staticmethod
    def from_cudf(df: cudf.DataFrame) -> LogicalTable: ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def get_column(self, column: int | str) -> LogicalColumn: ...
    def __getitem__(self, column: int | str) -> LogicalColumn: ...
    def get_column_names(self) -> List[str]: ...
    def to_array(self, *, out: Any) -> Any: ...
    def to_cudf(self) -> cudf.DataFrame: ...
    def offload_to(target_mem: StoreTarget) -> None: ...
    def repr(self, max_num_items: int) -> str: ...
    def __repr__(self) -> str: ...
