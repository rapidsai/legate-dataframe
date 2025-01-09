#!/usr/bin/env python

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


import argparse
import os.path
from pathlib import Path

from python_on_whales import docker

root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent

commands = {
    "cpp": [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "legate-dev",
        "ci/run_ctests.sh",
    ],
    "py": [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "legate-dev",
        "ci/run_pytests.sh",
    ],
}


def main(args):
    if not args.no_build:
        docker.build(
            root_dir,
            tags="legate-dataframe:latest",
        )
    for test in args.tests:
        print("*" * 100)
        print(f"run test {test}: {commands[test]}")
        docker.run(
            image="legate-dataframe:latest",
            command=commands[test],
            gpus="all",
            tty=True,
            # Low default leads to cryptic NCCL failures (with many gpus?):
            shm_size="2gb",
            envs={"LEGATE_TEST": "1"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and test Legate-Dataframe")

    parser.add_argument(
        "--no-build",
        action="store_true",
        default=False,
        help="Skip the docker build step",
    )
    parser.add_argument(
        "--tests",
        metavar="TEST",
        default=("cpp", "py"),
        nargs="+",
        choices=tuple(commands.keys()) + ("all",),
        help="List of tests to run {%(choices)s}",
    )

    args = parser.parse_args()
    if "all" in args.tests:
        args.tests = tuple(commands.keys())

    main(args)
