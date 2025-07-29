# Copyright (c) 2024-2025, NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Iterable, List, Tuple

import pyarrow as pa
import pytest

from legate_dataframe import LogicalTable
from legate_dataframe.lib.groupby_aggregation import groupby_aggregation
from legate_dataframe.testing import assert_arrow_table_equal


def arrow_groupby(
    table: pa.Table,
    keys: List[str],
    column_aggregations: Iterable[Tuple[str, str, str]],
) -> pa.Table:
    """Helper function that performs Arrow groupby using the legate syntax"""

    pyarrow_aggregations = [(a, b) for a, b, _ in column_aggregations]
    result = pa.TableGroupBy(table, keys).aggregate(pyarrow_aggregations)

    # rename the aggregations according to the given names
    names = keys.copy()

    for _, _, out_name in column_aggregations:
        names.append(out_name)
    return result.rename_columns(names)


@pytest.mark.parametrize(
    "keys,table",
    [
        (
            ["k1"],
            pa.table(
                {
                    "k1": ["x", "x", "y", "y", "z"],
                    "d1": [1, 2, 0, 4, 1],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
        (
            ["k1", "k2"],
            pa.table(
                {
                    "k1": ["x", "y", "y", "y", "x"],
                    "d1": [1, 2, 0, 4, 1],
                    "k2": ["y", "x", "y", "x", "y"],
                    "d2": [3, 2, 4, 5, 1],
                }
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        [("d1", "sum", "sum")],
        [("d1", "min", "min"), ("d1", "max", "max")],
        [("d2", "mean", "mean"), ("d1", "product", "prod")],
    ],
)
def test_aggregation(table, keys, aggs):
    expect = arrow_groupby(table, keys, aggs)

    tbl = LogicalTable.from_arrow(table)
    result = groupby_aggregation(tbl, keys, aggs)

    # sort before testing as the order of keys is arbitrary
    sort_keys = [(key, "ascending") for key in keys]
    assert_arrow_table_equal(result.sort_by(sort_keys), expect.sort_by(sort_keys), True)
