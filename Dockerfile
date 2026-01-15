# Copyright 2024 NVIDIA Corporation
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
#

ARG CUDA_VERSION="12.9.1"
ARG PYTHON_VERSION="3.11"

ARG BASE_IMAGE="rapidsai/ci-conda:cuda${CUDA_VERSION}-ubuntu24.04-py${PYTHON_VERSION}"
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Legate-dataframe: create conda environment, build, and install
#
# Update conda the environment (we use `cut` to convert the format of CUDA_VERSION from "12.9.1" to "129")
RUN mkdir -p /opt/legate-dataframe/conda-env-file
COPY ./conda/environments/*.yaml /opt/legate-dataframe/conda-env-file/

# To ensure we find the GPU version of legate in the docker build.
ARG CONDA_OVERRIDE_CUDA=12.9
RUN /bin/bash -c '/opt/conda/bin/mamba env create --name legate-dev --file \
  /opt/legate-dataframe/conda-env-file/all_cuda-$(cut --output-delimiter="" -d "." -f 1,2 <<< ${CONDA_OVERRIDE_CUDA})_arch-x86_64.yaml'

# Build and install legate-dataframe
WORKDIR /opt/legate-dataframe
COPY . /opt/legate-dataframe
ARG LEGATE_DF_BUILD_ARGS="-v liblegate_dataframe legate_dataframe"
RUN /bin/bash -c '\
  source activate legate-dev && \
  ./build.sh ${LEGATE_DF_BUILD_ARGS}'

# For convenience, activate legate-dev in docker interactive mode
RUN echo "conda activate legate-dev" >> ~/.bashrc
