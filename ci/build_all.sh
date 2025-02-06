#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -e -E -u -o pipefail

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
LEGATEDATAFRAME_PACKAGE_VERSION="$(head -1 ./VERSION)" \
rapids-conda-retry mambabuild \
    --channel legate \
    --channel legate/label/rc \
    --channel legate/label/branch-25.01 \
    --channel legate/label/experimental \
    --channel rapidsai \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-dataframe

# echo package details to logs, to help with debugging
conda search \
    --override-channels \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --info \
        legate-dataframe
