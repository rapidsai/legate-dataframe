# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from importlib import import_module
from pathlib import Path

import pytest

from legate_dataframe.testing import try_import_cudf

# skips tests if cudf is not available
cudf = try_import_cudf()

examples_path = Path(os.path.realpath(__file__)).parent / ".." / "examples"


def test_hello_world(tmp_path, monkeypatch):
    """Test examples/hello_world.py"""

    # hello_world.py imports cupynumeric
    pytest.importorskip("cupynumeric")

    monkeypatch.syspath_prepend(str(examples_path))
    import_module("hello_world").main(tmp_path)
