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
include_guard(GLOBAL)

find_package(Python REQUIRED COMPONENTS Development NumPy)

# Add pyarrow and numpy includes to each target
function(link_to_pyarrow_headers targets)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    ERROR_VARIABLE PYARROW_ERROR
    RESULT_VARIABLE PYARROW_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(${PYARROW_RESULT})
    message(FATAL_ERROR "Error while trying to obtain pyarrow include directory:\n${PYARROW_ERROR}")
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy as np; print(np.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    ERROR_VARIABLE NUMPY_ERROR
    RESULT_VARIABLE NUMPY_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(${NUMPY_RESULT})
    message(FATAL_ERROR "Error while trying to obtain numpy include directory:\n${NUMPY_ERROR}")
  endif()

  foreach(target IN LISTS targets)
    # PyArrow headers require numpy headers.
    target_include_directories(${target} PRIVATE "${NUMPY_INCLUDE_DIR}")
    target_include_directories(${target} PRIVATE "${PYARROW_INCLUDE_DIR}")
  endforeach()
endfunction()
