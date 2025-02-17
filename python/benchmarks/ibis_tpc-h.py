# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
from time import perf_counter_ns

# Use selected test queries as defined for ibis (not official)
# Query ten includes a limit, so skip for now.
from ibis.backends.tests.tpc.h.test_queries import test_01, test_05

from legate.core import get_legate_runtime
from legate_dataframe.experimental_ibis import LegateBackend


# Manually projected away columns until we implement auto-projection.
required_columns = {
    test_01: dict(
        lineitem=[
            "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_quantity",
            "l_extendedprice", "l_shipdate"]
    ),
    test_05: dict(
        customer=["c_custkey", "c_nationkey"],
        lineitem=["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"],
        orders=["o_custkey", "o_orderkey", "o_orderdate"],
        supplier=["s_suppkey", "s_nationkey"],
        region=["r_regionkey", "r_name"],
    ),
}


def run_test(test, file_glob):
    columns = required_columns[test]
    test = test.__wrapped__

    con = LegateBackend().connect()
    names = inspect.signature(test).parameters.keys()

    # Starting test (this reads data already)
    start_time = perf_counter_ns()
    args = (con.read_parquet(file_glob.format(name=name), columns=columns[name]) for name in names)
    # If the parquet files include an "ignore" column, drop it:
    args = (t.drop("ignore") if "ignore" in t.columns else t for t in args)
    # The test is nicely wrapped up, unwrap it
    result = test(*args)

    res = con.to_legate(result)
    get_legate_runtime().issue_execution_fence(block=True)
    return res, (perf_counter_ns() - start_time) / 1e9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join benchmark")
    parser.add_argument(
        "file_glob",
        help=("Glob pattern for the TPC-h parquet files after replacing `{name}` "
              "with the TPC-h table name (i.e. `lineitem`)."),
    )

    args = parser.parse_args()

    for test in (test_01, test_05):
        print("Running test", test)
        res, time = run_test(test, file_glob=args.file_glob)
        print("    Duration in seconds:", time)
