# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Iterable

import polars as plc
import pyarrow as pa
from numpy.typing import DTypeLike

from legate_dataframe.lib.core.table import LogicalTable

def csv_write(
    tbl: LogicalTable, path: pathlib.Path | str, delimiter: str = ","
) -> None: ...
def csv_read(
    files: pathlib.Path | str | Iterable[pathlib.Path | str],
    *,
    dtypes: Iterable[DTypeLike | plc.DataType | pa.DataType],
    na_filter: bool = False,
    delimiter: str = ",",
    usecols: Iterable[str | int] | None = None,
    names: Iterable[str] | None = None,
) -> LogicalTable: ...
