# Copyright 2024 NVIDIA Corporation
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

import argparse
import contextlib
import functools
import statistics
import time
from typing import Tuple

import cudf
import numpy
from cudf.utils.string import format_bytes


def create_key_and_data(args, module):
    if "dask" in args.api:
        # Dask doesn't support argsort and random.permutation is very slow,
        # instead we use an (very good) approximation of --unique-factor.
        key = module.random.random_integers(low=0, high=2**50, size=args.nrows)
    else:
        key = module.argsort(module.random.random(size=args.nrows))
    if args.unique_factor > 0:
        key %= int(args.nrows * args.unique_factor)
    data = module.arange(args.nrows)
    return key.astype(args.dtype), data.astype(args.dtype)


@contextlib.contextmanager
def run_dask(args, *, use_gpu):
    import dask.config
    from dask import array as da
    from dask import dataframe as dd
    from dask_cuda import LocalCUDACluster
    from distributed import Client, wait

    dask.config.set(explicit_comms=args.dask_explicit_comms)

    def create_table(args, name: str) -> dd.DataFrame:
        key, data = create_key_and_data(args, module=da)
        ret: dd.DataFrame = dd.from_dask_array(key, columns=f"{name}-key").to_frame()
        ret[f"{name}-data"] = dd.from_dask_array(data)
        if use_gpu:
            ret = ret.map_partitions(cudf.from_pandas)
        ret = ret.reset_index(drop=True)
        return ret.repartition(npartitions=args.dask_n_workers)

    with LocalCUDACluster(
        n_workers=args.dask_n_workers,
        protocol=args.dask_protocol,
        rmm_pool_size=args.dask_rmm_pool,
    ) as cluster:
        with Client(cluster):

            def f():
                lhs = create_table(args, name="lhs").persist()
                rhs = create_table(args, name="rhs").persist()
                wait([lhs, rhs])
                t0 = time.perf_counter()
                res = dd.merge(
                    lhs,
                    rhs,
                    left_on=["lhs-key"],
                    right_on=["rhs-key"],
                    how="inner",
                ).persist()
                wait([res])
                t1 = time.perf_counter()
                return t1 - t0, len(res)

            yield f


@contextlib.contextmanager
def run_legate(args):
    import cupynumeric
    from legate.core import get_legate_runtime

    from legate_dataframe import LogicalColumn, LogicalTable
    from legate_dataframe.lib.join import JoinType, join, null_equality

    runtime = get_legate_runtime()

    def blocking_timing() -> float:
        runtime.issue_execution_fence(block=True)
        return time.perf_counter()

    def create_table(args, name: str) -> LogicalTable:
        key, data = create_key_and_data(args, module=cupynumeric)
        return LogicalTable(
            columns=(LogicalColumn(key.astype(args.dtype)), LogicalColumn(data)),
            column_names=(f"{name}-key", f"{name}-data"),
        )

    def f() -> Tuple[float, int]:
        lhs = create_table(args, name="lhs")
        rhs = create_table(args, name="rhs")
        t0 = blocking_timing()
        res = join(
            lhs,
            rhs,
            lhs_keys=["lhs-key"],
            rhs_keys=["rhs-key"],
            join_type=JoinType.INNER,
            compare_nulls=null_equality.EQUAL,
            _num_paritions=args.legate_parts,
        )
        t1 = blocking_timing()
        return t1 - t0, res.num_rows()

    yield f


API = {
    "legate": run_legate,
    "dask-cpu": functools.partial(run_dask, use_gpu=False),
    "dask-gpu": functools.partial(run_dask, use_gpu=True),
}


def main(args):
    ncols = 4  # total number of columns in the two input tables and the output table
    itemsize = numpy.dtype(args.dtype).itemsize
    nbytes_input = args.nrows * ncols * itemsize

    print("==================================================")
    print("Inner Join Benchmark")
    print("--------------------------------------------------")
    print(f"api                 | {args.api}")
    print(f"nrows               | {args.nrows} ({format_bytes(nbytes_input)})")
    print(f"dtype               | {args.dtype}")
    print(f"unique-factor       | {args.unique_factor}")
    print(f"nruns               | {args.nruns}")
    print(f"nwarms              | {args.nwarms}")
    if "dask" in args.api:
        print(f"dask-n-workers      | {args.dask_n_workers}")
        print(f"dask-protocol       | {args.dask_protocol}")
        print(f"dask-explicit-comms | {args.dask_explicit_comms}")
        print(f"dask-rmm-pool       | {args.dask_rmm_pool}")

    print("==================================================")

    timings = []
    nbytes_total = 0
    with API[args.api](args) as f:
        for i in range(args.nwarms):
            elapsed, num_out_rows = f()
            nbytes = num_out_rows * ncols * itemsize + nbytes_input  # Total size
            print(
                f"elapsed[warm #{i}]: {elapsed:.4f}s ({nbytes/elapsed/2**30:.3f} GiB/s)"
            )
        for i in range(args.nruns):
            elapsed, num_out_rows = f()
            nbytes = num_out_rows * ncols * itemsize + nbytes_input  # Total size
            print(
                f"elapsed[run #{i}]: {elapsed:.4f}s ({nbytes/elapsed/2**30:.3f} GiB/s)"
            )
            nbytes_total += nbytes
            timings.append(elapsed)
    print("--------------------------------------------------")
    print(
        f"mean: {statistics.mean(timings):.4}s "
        f"± {(statistics.stdev(timings) if len(timings) > 1 else 0.0):.4}s "
        f"({nbytes_total/sum(timings)/2**30:.3f} GiB/s)"
    )
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join benchmark")
    parser.add_argument(
        "--api",
        metavar="API",
        default=tuple(API.keys())[0],
        choices=tuple(API.keys()),
        help="API to use {%(choices)s}",
    )
    parser.add_argument(
        "--nrows",
        default="1e6",
        type=lambda x: int(float(x)),
        help="Number of rows in each table (default: %(default)s).",
    )
    parser.add_argument(
        "--nruns",
        default=1,
        type=int,
        help="Number of runs (default: %(default)s).",
    )
    parser.add_argument(
        "--nwarms",
        default=0,
        type=int,
        help="Number of warm-up runs (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        type=str,
        help="Datatype (default: %(default)s).",
    )
    parser.add_argument(
        "--unique-factor",
        default=1.0,
        type=float,
        help=(
            "The ratio of unique rows in the key columns, which is a value between "
            '1 and 0 where 1 is "all rows are unique" and 0 is "all rows are the same" '
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--dask-n-workers",
        metavar="N_WORKERS",
        default=1,
        type=int,
        help="Number of workers (default: %(default)s).",
    )
    parser.add_argument(
        "--dask-protocol",
        metavar="PROTOCOL",
        default="tcp",
        type=str,
        help="The protocol to use (default: %(default)s).",
    )
    parser.add_argument(
        "--dask-explicit-comms",
        metavar="EXPLICIT_COMMS",
        default=True,
        action=argparse.BooleanOptionalAction,
        type=bool,
        help="Whether to enable Dask-CUDA's explicit-comms or not.",
    )
    parser.add_argument(
        "--dask-rmm-pool",
        metavar="RMM_POOL_SIZE",
        default=None,
        help="The initial RMM pool size e.g. 1GiB (default: disabled).",
    )
    parser.add_argument(
        "--legate-parts",
        default=-1,
        type=int,
        help="Use legate partitioning approach (and how many partitions).",
    )
    args = parser.parse_args()
    main(args)
