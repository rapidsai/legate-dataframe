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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# read project version from VERSION file
file(READ "${CMAKE_CURRENT_LIST_DIR}/legate_dataframe/VERSION" _version_content)
if(_version_content MATCHES [[^([0-9]+)\.([0-9]+)\.([0-9]+)]])
  set(_py_legate_dataframe_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _py_legate_dataframe_content_formatted "  ${_version_content}")
  message(
    FATAL_ERROR
      "Could not determine project version. Contents of VERSION file:\n${_py_legate_dataframe_content_formatted}"
  )
endif()

include(../cpp/cmake/fetch_rapids.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(PyLegateDataframe)

project(
  PyLegateDataframe
  VERSION ${_py_legate_dataframe_content_formatted}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

option(
  FIND_LegateDataframe
  "Search for existing legate dataframe C++ installations before defaulting to local files" OFF
)
option(LEGATE_DATAFRAME_USE_CUDA "Build library with gpu_tasks" ON)
if(LEGATE_DATAFRAME_USE_CUDA)
  add_definitions(-DLEGATE_DATAFRAME_USE_CUDA)
endif()

if(FIND_LegateDataframe)
  find_package(LegateDataframe ${PyLegateDataframe_version})
else()
  set(LegateDataframe_FOUND OFF)
endif()

if(NOT LegateDataframe_FOUND)
  add_subdirectory(../cpp liblegate_dataframe)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(legate_dataframe/lib)
add_subdirectory(legate_dataframe/lib/core)
