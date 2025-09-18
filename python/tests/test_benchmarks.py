# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

benchmarks_path = Path(os.path.realpath(__file__)).parent / ".." / "benchmarks"


def test_join(tmp_path, monkeypatch):
    """Test examples/hello_world.py"""

    # join.py imports cupynumeric/cudf
    pytest.importorskip("cupynumeric")
    pytest.importorskip("cudf")

    monkeypatch.syspath_prepend(str(benchmarks_path))
    args = SimpleNamespace()
    args.api = "legate"
    args.nrows = 2**20
    args.nruns = 1
    args.nwarms = 1
    args.dtype = "float64"
    args.unique_factor = 1.0
    import_module("join").main(args)
