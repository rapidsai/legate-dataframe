# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Iterable, Optional

from legate_dataframe.lib.core.table import LogicalTable

class null_equality(Enum):
    EQUAL: int
    UNEQUAL: int

class JoinType(Enum):
    INNER: int
    LEFT: int
    FULL: int

class BroadcastInput(Enum):
    AUTO: int
    LEFT: int
    RIGHT: int

def join(
    lhs: LogicalTable,
    rhs: LogicalTable,
    *,
    lhs_keys: Iterable[str],
    rhs_keys: Iterable[str],
    join_type: JoinType,
    lhs_out_columns: Optional[Iterable[str]] = None,
    rhs_out_columns: Optional[Iterable[str]] = None,
    nulls_equal: bool = True,
    broadcast: BroadcastInput = BroadcastInput.AUTO,
) -> LogicalTable: ...
