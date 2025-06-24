# Copyright (c) 2023-2024: int NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn
from legate_dataframe.lib.core.scalar import ScalarLike

def binary_operation(
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
    op: str,
    output_type: DTypeLike,
) -> LogicalColumn: ...
