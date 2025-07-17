# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.scalar import ScalarLike

def reduce(
    col: LogicalColumn,
    op: str,
    output_type: DTypeLike,
    *,
    initial: ScalarLike | None = None,
) -> LogicalColumn: ...
