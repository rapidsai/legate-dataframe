# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import pytest
from legate.core import StoreTarget, get_legate_runtime

from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.stream_compaction import apply_boolean_mask
from legate_dataframe.testing import guess_available_mem


def test_offload_to():
    # Note that, if `LEGATE_CONFIG` is set but not used, this may currently fail.
    available_mem_gpu, available_mem_cpu = guess_available_mem()
    if not available_mem_gpu or not available_mem_cpu:
        pytest.skip(reason="Could not guess available GPU or SYSMEM.")
    if available_mem_cpu < available_mem_gpu * 2.5:
        pytest.skip(reason="Need a more SYSMEM than GPU mem for test.")

    length = available_mem_gpu // 10 * 1024**2
    col = cudf.Series([True], dtype="bool")
    col = col.repeat(length)
    col_lg = LogicalColumn.from_cudf(col._column)
    tbl = LogicalTable([col_lg], "a")

    results = []
    for i in range(15):
        # Taking the negative 15 times can't possibly fit into GPU memory
        res = apply_boolean_mask(tbl, col_lg)
        # but should work if we offload all results
        res.offload_to(StoreTarget.SYSMEM)
        results.append(res)

        # Make sure we clean up before we continue (or finalize the program)
        # (As of writing, doing it every time prevents a hang.)
        get_legate_runtime().issue_execution_fence(block=True)

    # Not sure if helpful, but delete and wait.
    del col_lg, tbl, results
    get_legate_runtime().issue_execution_fence(block=True)
