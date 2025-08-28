# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple

from legate_dataframe.lib.core.table import LogicalTable

def groupby_aggregation(
    table: LogicalTable,
    keys: Iterable[str],
    column_aggregations: Iterable[Tuple[str | None, str, str]],
) -> LogicalTable: ...
