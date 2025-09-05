# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from legate_dataframe.lib.core.column import LogicalColumn

_MatchFuncs = Literal[
    "starts_with", "ends_with", "match_substring", "match_substring_regex"
]

def match(
    match_func: _MatchFuncs, column: LogicalColumn, pattern: str
) -> LogicalColumn: ...
