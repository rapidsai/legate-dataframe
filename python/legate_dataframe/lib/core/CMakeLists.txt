# =============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Set the list of Cython files to build

if(Legion_USE_CUDA)
  file(
    GLOB CYTHON_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.pyx"
  )
else()
  file(
    GLOB CYTHON_SOURCES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*[!_cu].pyx"
  )
endif()

# rapids_cython_create_modules passes CYTHON_FLAGS to the cython command line
if(Legion_USE_CUDA)
  set(CYTHON_FLAGS "${CYTHON_FLAGS} -E LEGATE_DATAFRAME_USE_CUDA=1")
else()
  set(CYTHON_FLAGS "${CYTHON_FLAGS} -E LEGATE_DATAFRAME_USE_CUDA=0")
endif()

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${CYTHON_SOURCES}"
  LINKED_LIBRARIES LegateDataframe::LegateDataframe
)

# TODO: Vendored the file from cudf, can probably make it simpler here?
include(../../cmake/Modules/LinkPyarrowHeaders.cmake)
link_to_pyarrow_headers("${RAPIDS_CYTHON_CREATED_TARGETS}")
