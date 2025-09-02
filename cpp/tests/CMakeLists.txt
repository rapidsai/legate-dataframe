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

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

include(rapids-test)
# NB: calling `rapids_test_init()` to have CTest manage GPU resource doesn't work

file(GLOB SOURCES "*.cpp")
if(LEGATE_DATAFRAME_USE_CUDA)
  file(GLOB CUDA_SOURCES "*.cu")
  list(APPEND SOURCES ${CUDA_SOURCES})
endif()

add_executable(cpp_tests ${SOURCES})
set_target_properties(
  cpp_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${LegateDataframe_BINARY_DIR}/gtests>"
             # INSTALL_RPATH "\$ORIGIN/../../../lib"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
)

set(LDF_TEST_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
target_compile_options(cpp_tests PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${LDF_TEST_CUDA_FLAGS}>")

target_link_libraries(
  cpp_tests PRIVATE LegateDataframe GTest::gmock Arrow::arrow_shared ArrowAcero::arrow_acero_shared
                    GTest::gtest $<TARGET_NAME_IF_EXISTS:conda_env>
)
if(LEGATE_DATAFRAME_USE_CUDA)
  target_link_libraries(
    cpp_tests PRIVATE LegateDataframe cudf cudf::cudftestutil cudf::cudftestutil_impl
  )
endif()

rapids_test_add(
  NAME cpp_tests
  COMMAND cpp_tests
  INSTALL_COMPONENT_SET testing
)

# ##################################################################################################
# Install tests ------------------------------------------------------------------------------------
# ##################################################################################################
rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/liblegate_dataframe
)
