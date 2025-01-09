#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

# [description]
#
#   Run (GPU) tests.
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

# Unless otherwise specified, use all available GPUs and set
# LEGATE_TEST=1 to test broadcasting code paths (locally).
# TODO: Set LEGATE_CONFIG instead (if undefined).  However,
#       as of 2024-10-11 LEGATE_CONFIG seems broken:
#       https://github.com/nv-legate/legate.core.internal/issues/1304
LEGATE_TEST=${LEGATE_TEST:-1} \
legate \
    --gpus "$(nvidia-smi -L | wc -l)"\
    --fbmem=4000 \
    --module pytest \
    . \
    -sv \
    --durations=0 \
    "${@}"
