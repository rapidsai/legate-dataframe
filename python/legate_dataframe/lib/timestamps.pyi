# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.column import LogicalColumn

__all__ = ["to_timestamps", "extract_timestamp_component"]

def to_timestamps(
    col: LogicalColumn,
    timestamp_type: DTypeLike,
    format_pattern: str,
) -> LogicalColumn: ...
def extract_timestamp_component(
    col: LogicalColumn,
    component: str,
) -> LogicalColumn: ...
