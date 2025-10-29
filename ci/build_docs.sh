#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -e -E -u -o pipefail

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-generate-version > ./VERSION
LEGATEDATAFRAME_VERSION=$(rapids-version)

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

rapids-mamba-retry env create \
    --yes \
    --file /tmp/env.yaml \
    --name docs-env

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs-env
set -u

rapids-print-env

# Install legate-dataframe conda package built in the previous CI job
rapids-mamba-retry install \
  --name docs-env \
  --override-channels \
  --channel "${RAPIDS_LOCAL_CONDA_CHANNEL}" \
  --channel legate \
  --channel legate/label/rc \
  --channel legate-nightly \
  --channel rapidsai \
  --channel conda-forge \
    "legate-dataframe=${LEGATEDATAFRAME_VERSION}"

rapids-print-env

# Make documentation, we need gpu but otherwise minimal resources
LEGATE_CONFIG="--omps=0 --gpus=1 --fbmem=100 --cpus=1 --utility=1" \
make -C docs html
