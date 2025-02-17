# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Iterable, Tuple

from legate_dataframe.lib.core.table import LogicalTable

class AggregationKind(Enum):
    SUM: int
    PRODUCT: int
    MIN: int
    MAX: int
    COUNT: int
    SIZE: int
    ANY: int
    ALL: int
    SUM_OF_SQUARES: int
    MEAN: int
    VAR: int
    STD: int
    MEDIAN: int
    QUANTILE: int
    ARGMAX: int
    ARGMIN: int
    NUNIQUE: int
    NTH: int
    RANK: int
    COLLECT: int
    UNIQUE: int
    PTX: int
    CUDA: int
    CORRELATION: int
    COVARIANCE: int
    COUNT_ALL: int
    COUNT_VALID: int

def groupby_aggregation(
    table: LogicalTable,
    keys: Iterable[str],
    column_aggregations: Iterable[Tuple[str, AggregationKind, str]],
) -> LogicalTable: ...
