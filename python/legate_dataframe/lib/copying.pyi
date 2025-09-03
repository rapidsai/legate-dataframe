# Copyright (c) 2023-2024: int NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable

from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.scalar import ScalarLike

def copy_if_else(
    cond: LogicalColumn,
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
) -> LogicalColumn: ...
def concatenate(
    columns: Iterable[LogicalColumn],
) -> LogicalColumn: ...
