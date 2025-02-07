#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# [description]
#
# Common environment-setup stuff used by different CI jobs that
# test conda packages.
#
# This is intended to be source'd by other test scripts.

set -e -E -u -o pipefail

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-generate-version > ./VERSION
LEGATEDATAFRAME_VERSION="$(head -1 ./VERSION)"

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key test \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

rapids-mamba-retry env create \
    --yes \
    --file /tmp/env.yaml \
    --name test-env

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test-env
set -u

rapids-print-env

# Install legate-dataframe conda package built in the previous CI job
rapids-mamba-retry install \
  --name test-env \
  --channel "${RAPIDS_LOCAL_CONDA_CHANNEL}" \
  --channel legate \
  --channel legate/label/rc \
  --channel legate/label/branch-25.01 \
  --channel legate/label/experimental \
  --channel rapidsai \
  --channel conda-forge \
  --channel nvidia \
    "legate-dataframe=${LEGATEDATAFRAME_VERSION}" \
    "legate-df-ctests=${LEGATEDATAFRAME_VERSION}"
