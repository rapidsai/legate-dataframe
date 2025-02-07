# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import pathlib
from typing import Iterable

from legate_dataframe.lib.core.table import LogicalTable

def parquet_write(tbl: LogicalTable, path: pathlib.Path | str) -> None: ...
def parquet_read(
    glob_string: pathlib.Path | str, *, columns: Iterable[str] | None
) -> LogicalTable: ...
