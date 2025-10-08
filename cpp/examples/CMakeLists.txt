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

set(TEST_INSTALL_PATH bin/tests/hello_world)
set(TEST_NAME hello_world)

add_executable(hello_world hello.cpp)
set_target_properties(hello_world PROPERTIES INSTALL_RPATH "\$ORIGIN/.." CXX_STANDARD 20)
target_link_libraries(hello_world PRIVATE LegateDataframe::LegateDataframe)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(LegateDataframe_CXX_FLAGS "-Wall;-Werror;-Wno-unknown-pragmas")
  target_compile_options(
    hello_world PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LegateDataframe_CXX_FLAGS}>"
  )
endif()

install(
  TARGETS ${TEST_NAME}
  COMPONENT examples
  DESTINATION ${TEST_INSTALL_PATH}
  EXCLUDE_FROM_ALL
)
