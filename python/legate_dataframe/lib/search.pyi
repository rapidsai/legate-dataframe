# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .core.column import LogicalColumn

def contains(haystack: LogicalColumn, needles: LogicalColumn) -> LogicalColumn: ...
