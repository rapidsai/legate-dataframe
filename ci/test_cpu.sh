#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -e -E -u -o pipefail

# Ensure this is running from the root of the repo
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../;

# Common setup steps shared by Python test jobs (as of writing, just one)
source ./ci/test_cpu_common.sh

nvidia-smi

rapids-logger "Running C++ tests"
./ci/run_ctests_cpu.sh

# run the tests
rapids-logger "Running Python tests"
./ci/run_pytests_cpu.sh
