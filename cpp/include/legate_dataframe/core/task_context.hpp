/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <tuple>

#include <legate.h>

#ifdef LEGATE_DATAFRAME_USE_CUDA
#include <rmm/cuda_stream.hpp>

#include <legate_dataframe/core/allocator.cuh>
#endif

namespace legate {
namespace dataframe {

/**
 * @brief This helper class is to make sure that each GPU task uses its own
 * allocator for temporary allocations from libcudf during its execution.
 * This class also creates a fresh stream to be used for kernels.
 * Do not call `cudaMalloc()` or `cudaMallocAsync()`, it might deadlock!
 * This is because they might block the _whole_ device. Instead use the
 * task's RMM resource `.mr()` or `legate::create_buffer()`.
 */
class TaskContext {
 protected:
  legate::TaskContext& _context;
  size_t _arg_scalar_idx{0};
  size_t _arg_input_idx{0};
  size_t _arg_output_idx{0};
#ifdef LEGATE_DATAFRAME_USE_CUDA
  std::optional<TaskMemoryResource> _mr;
#endif

 public:
  const int rank;
  const int nranks;

  /**
   * @brief Create a new task context from a legate task context
   *
   * @param context The legate task context
   * @param arg_scalar_idx The initial task argument index for scalars. If this task was launched
   * with scalar arguments unknown to legate-dataframe (e.g. by using `AutoTask::add_scalar_arg()`),
   * set this to the number of such scalar arguments used.
   * @param arg_input_idx The initial task argument index for inputs. If this task was launched
   * with input arguments unknown to legate-dataframe (e.g. by using `AutoTask::add_input()`), set
   * this to the number of such input arguments used.
   * @param arg_output_idx The initial task argument index for outputs. If this task was launched
   * with output arguments unknown to legate-dataframe (e.g. by using `AutoTask::add_output()`),
   * set this to the number of such output arguments used.
   */
  TaskContext(legate::TaskContext& context,
              size_t arg_scalar_idx = 0,
              size_t arg_input_idx  = 0,
              size_t arg_output_idx = 0);

  /**
   * @brief Get the current indices of the task arguments
   *
   * When mixing the use of legate-dataframe and legate.core task arguments, use this function
   * to get the task argument indices. For example, after the legate-dataframe arguments have been
   * read using `argument::get_next_scalar()`, `::get_next_input()`, and `::get_next_output()`, a
   * task can use this function to get the initial indices for `legate::TaskContext::scalars()`,
   * `::input()`,  and `::output()`.
   *
   * @return A tuple of `(scalar, input, output)` indices.
   */
  std::tuple<size_t, size_t, size_t> get_task_argument_indices() const
  {
    return {_arg_scalar_idx, _arg_input_idx, _arg_output_idx};
  }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  TaskMemoryResource* mr()
  {
    if (!_mr.has_value()) { _mr = TaskMemoryResource(); }
    return &_mr.value();
  }

  /**
   * @brief Get the CUDA stream to be used in this task context.
   *
   * All legate tasks should use this stream.  The function is mainly for
   * convenience to mirror easy access to `mr()`.
   *
   * @return The tasks CUDA stream
   */
  cudaStream_t stream() { return _context.get_task_stream(); }
#endif

  legate::Scalar get_next_scalar_arg() { return _context.scalars().at(_arg_scalar_idx++); }
  legate::PhysicalArray get_next_input_arg() { return _context.input(_arg_input_idx++); }
  legate::PhysicalArray get_next_output_arg() { return _context.output(_arg_output_idx++); }

  const legate::TaskContext& get_legate_context() const { return _context; }
};

}  // namespace dataframe
}  // namespace legate
