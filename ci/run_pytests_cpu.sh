#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

# [description]
#
#   Run (CPU) tests.
#
#   This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   Put CI-specific details in 'test_python_*.sh'.
#
#   Additional arguments passed to this script are passed through to 'pytest'.
#

set -e -E -u -o pipefail

# Support invoking run_cudf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/tests/

# Unless `LEGATE_CONFIG` is set, default to all available GPUs and set fbmem/sysmem.
# The choice of 2000 and 6000 allows some large memory tests to run (on a single GPU).
# LEGATE_TEST=1 to test broadcasting code paths (locally).
LEGATE_CONFIG=${LEGATE_CONFIG:- --cpus 8 --gpus 0 --sysmem=6000 --omps=0} \
LEGATE_TEST=${LEGATE_TEST:-1} \
legate \
    --module pytest \
    . \
    -sv \
    --durations=0 \
    -k 'csv or binary or parquet' \
    "${@}"
