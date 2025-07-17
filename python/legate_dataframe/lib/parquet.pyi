# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import pathlib
from typing import Iterable

import legate.core

from legate_dataframe.lib.core.table import LogicalTable

def parquet_write(tbl: LogicalTable, path: pathlib.Path | str) -> None: ...
def parquet_read(
    files: pathlib.Path | str | Iterable[pathlib.Path | str],
    *,
    columns: Iterable[str] | None,
) -> LogicalTable: ...
def parquet_read_array(
    files: pathlib.Path | str | Iterable[pathlib.Path | str],
    *,
    columns: Iterable[str] | None = None,
    null_value: legate.core.Scalar | None = None,
    type: legate.core.Type | None = None,
) -> legate.core.LogicalArray: ...
