# =============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

include(cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

# read project version from VERSION file
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _version_content)
if(_version_content MATCHES [[^([0-9]+)\.([0-9]+)\.([0-9]+)]])
  set(_legate_dataframe_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _legate_dataframe_content_formatted "  ${_version_content}")
  message(
    FATAL_ERROR
      "Could not determine project version. Contents of VERSION file:\n${_legate_dataframe_content_formatted}"
  )
endif()

project(
  LegateDataframe
  VERSION "${_legate_dataframe_version}"
  LANGUAGES CXX CUDA
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build legate-dataframe shared library" ON)
option(BUILD_EXAMPLES "Configure CMake to build examples" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(LEGATE_DATAFRAME_USE_CUDA "Build library with gpu_tasks" ON)
if(LEGATE_DATAFRAME_USE_CUDA)
  add_definitions(-DLEGATE_DATAFRAME_USE_CUDA)
endif()

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(LegateDataframe_BUILD_EXAMPLES ${BUILD_EXAMPLES})
set(LegateDataframe_BUILD_TESTS ${BUILD_TESTS})

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

if(LEGATE_DATAFRAME_USE_CUDA)
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET LegateDataframe-exports
    INSTALL_EXPORT_SET LegateDataframe-exports
  )
  include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags
endif()

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(LEGATE_DATAFRAME_USE_CUDA)
  # If CCCL versions don't get picked up correctly, we may need to add them early on here (right now
  # it seems fetching rmm/cudf manage to ensure we ge the right one.)
  include("${rapids-cmake-dir}/cpm/cccl.cmake")
  rapids_cpm_cccl()
endif()

if(LEGATE_DATAFRAME_USE_CUDA)
  include(rapids-cuda)
  rapids_cuda_init_architectures(LegateDataframe)
  include(cmake/thirdparty/get_rmm.cmake)
  include(cmake/thirdparty/get_cudf.cmake)
  rapids_find_generate_module(
    NCCL
    HEADER_NAMES nccl.h
    LIBRARY_NAMES nccl
  )

  # Currently NCCL has no CMake build-system so we require it built and installed on the machine
  # already
  rapids_find_package(NCCL REQUIRED)
endif()

rapids_find_package(
  legate REQUIRED Legion LegionRuntime
  BUILD_EXPORT_SET LegateDataframe-exports
  INSTALL_EXPORT_SET LegateDataframe-exports
)

rapids_find_package(
  Arrow REQUIRED
  BUILD_EXPORT_SET LegateDataframe-exports
  INSTALL_EXPORT_SET LegateDataframe-exports
)

rapids_find_package(
  ArrowAcero REQUIRED
  BUILD_EXPORT_SET LegateDataframe-exports
  INSTALL_EXPORT_SET LegateDataframe-exports
)

rapids_find_package(
  Parquet REQUIRED
  BUILD_EXPORT_SET LegateDataframe-exports
  INSTALL_EXPORT_SET LegateDataframe-exports
)

if(LegateDataframe_BUILD_TESTS)
  # find or install GoogleTest
  include(cmake/thirdparty/get_gtest.cmake)
endif()

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

if(LEGATE_DATAFRAME_USE_CUDA)
  file(GLOB SOURCES "src/*.cpp" "src/*.cu" "src/core/*.cpp" "src/core/*.cu")
else()
  file(GLOB SOURCES "src/*.cpp" "src/core/*.cpp")
endif()

add_library(LegateDataframe ${SOURCES})
add_library(LegateDataframe::LegateDataframe ALIAS LegateDataframe)

set_target_properties(
  LegateDataframe
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

if(LEGATE_DATAFRAME_USE_CUDA)
  set_target_properties(LegateDataframe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

# Need to add this define, as is done in legate CMakeLists.txt as well for __half support.  If
# __half support fails elsewhere it may be needed there. This may be a CCCL 2.7.0 issue and become
# unnecessary in the future.
target_compile_definitions("LegateDataframe" PUBLIC _LIBCUDACXX_HAS_NVFP16=1)

list(APPEND LDF_CUDA_FLAGS --expt-extended-lambda)
list(APPEND LDF_CUDA_FLAGS --expt-relaxed-constexpr)

target_compile_options(
  LegateDataframe PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LDF_CXX_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${LDF_CUDA_FLAGS}>"
)

target_include_directories(
  LegateDataframe
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(
  LegateDataframe
  PUBLIC legate::legate Arrow::arrow_shared
  PRIVATE ArrowAcero::arrow_acero_shared Parquet::parquet_shared
)
if(LEGATE_DATAFRAME_USE_CUDA)
  target_link_libraries(
    LegateDataframe
    PUBLIC rmm::rmm cudf::cudf
    PRIVATE NCCL::NCCL
  )
endif()

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(LegateDataframe PRIVATE conda_env)
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(LegateDataframe_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # ctest cuda memcheck
  find_program(CUDA_SANITIZER compute-sanitizer)
  set(MEMORYCHECK_COMMAND ${CUDA_SANITIZER})
  set(MEMORYCHECK_TYPE CudaSanitizer)
  set(CUDA_SANITIZER_COMMAND_OPTIONS "--tool memcheck")

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add examples ----------------------------------------------------------------------------------

if(LegateDataframe_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------

rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME LegateDataframe)

install(
  TARGETS LegateDataframe
  DESTINATION ${lib_dir}
  EXPORT LegateDataframe-exports
)
install(DIRECTORY include/legate_dataframe/ DESTINATION include/legate_dataframe)

set(doc_string
    [=[
Legate-dataframe: a scalable dataframe library.
]=]
)

rapids_export(
  INSTALL LegateDataframe
  EXPORT_SET LegateDataframe-exports
  GLOBAL_TARGETS LegateDataframe
  NAMESPACE LegateDataframe::
  DOCUMENTATION doc_string
)
