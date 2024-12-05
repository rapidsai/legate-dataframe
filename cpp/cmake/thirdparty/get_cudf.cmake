# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

# This function finds cudf and sets any additional necessary environment variables.
function(find_and_configure_cudf)

  if(TARGET cudf::cudf)
    return()
  endif()

  set(oneValueArgs VERSION GIT_REPO GIT_TAG USE_CUDF_STATIC EXCLUDE_FROM_ALL
                   PER_THREAD_DEFAULT_STREAM
  )
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(global_targets cudf::cudf)
  set(cudf_components "")

  if(BUILD_TESTS OR BUILD_BENCHMARKS)
    list(APPEND global_targets cudf::cudftestutil)
    set(cudf_components COMPONENTS testing)
  endif()

  rapids_cpm_find(
    cudf ${PKG_VERSION} ${cudf_components}
    GLOBAL_TARGETS ${global_targets}
    BUILD_EXPORT_SET LegateDataframe-exports
    INSTALL_EXPORT_SET LegateDataframe-exports
    CPM_ARGS
    GIT_REPOSITORY ${PKG_GIT_REPO}
    GIT_TAG ${PKG_GIT_TAG}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF" "BUILD_SHARED_LIBS ON"
            "CUDF_BUILD_TESTUTIL ${BUILD_TESTS}" "CUDF_BUILD_STREAMS_TEST_UTIL OFF"
  )

  if(TARGET cudf)
    set_property(TARGET cudf PROPERTY SYSTEM TRUE)
  endif()
endfunction()

find_and_configure_cudf(
  VERSION 24.10 GIT_REPO https://github.com/rapidsai/cudf.git GIT_TAG branch-24.10
)
