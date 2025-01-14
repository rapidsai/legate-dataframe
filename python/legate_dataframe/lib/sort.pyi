# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from legate_dataframe.lib.core.table import LogicalTable

def sort(
    tbl: LogicalTable,
    keys: Iterable[str],
    *,
    stable: bool,
) -> LogicalTable: ...
