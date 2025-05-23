#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location.  Use the build dir if
# it seems to exist, otherwise assume we installed the tests.
if [[ -f "${LIBLEGATE_DATAFRAME_BUILD_DIR:-./cpp/build}/gtests/cpp_tests" ]]; then
    cd "${LIBLEGATE_DATAFRAME_BUILD_DIR:-./cpp/build}/gtests"
else
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/liblegate_dataframe/"
fi

# Unless `LEGATE_CONFIG` is set, default to all available GPUs and set fbmem/sysmem.
# LEGATE_TEST=1 to test broadcasting code paths (locally).
LEGATE_CONFIG=${LEGATE_CONFIG:- --gpus="$(nvidia-smi -L | wc -l)" --fbmem=2000 --sysmem=6000 --omps=0} \
LEGATE_TEST=${LEGATE_TEST:-1} \
legate ./cpp_tests --output-on-failure --no-tests=error "$@"
