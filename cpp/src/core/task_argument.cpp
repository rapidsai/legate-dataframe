/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <string>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/task_argument.hpp>

namespace legate::dataframe::argument {

void add_alignment_constraints(legate::AutoTask& task,
                               const std::vector<legate::Variable>& variables)
{
  for (size_t i = 1; i < variables.size(); ++i) {
    task.add_constraint(legate::align(variables[0], variables[i]));
  }
}

namespace {
size_t get_min_gpu_chunk()
{
  const char* env_p = std::getenv("LEGATE_MIN_GPU_CHUNK");
  if (env_p == nullptr) { env_p = "1048576"; }
  return std::stoi(env_p);
}
}  // namespace

void add_parallel_launch_task(legate::AutoTask& task, int min_num_tasks)
{
  static size_t min_gpu_chunk = get_min_gpu_chunk();
  auto runtime                = legate::Runtime::get_runtime();

  // TODO: in order to force a parallel launch, we send a dummy column along.
  //       When `legate::ManualTask` support `legate::LogicalArray`, we should use a
  //       manual task instead.
  auto dummy_fill  = legate::Scalar(0);
  auto dummy_array = runtime->create_array({min_gpu_chunk * min_num_tasks}, dummy_fill.type());
  runtime->issue_fill(dummy_array, dummy_fill);
  LogicalColumn dummy_column{dummy_array};
  add_next_input(task, dummy_column);
}

void add_parallel_launch_task(legate::AutoTask& task)
{
  // This should be the number of "preferred" processors
  auto num_processors = legate::Runtime::get_runtime()->get_machine().count();
  add_parallel_launch_task(task, num_processors);
}

void get_parallel_launch_task(TaskContext& ctx) { get_next_input<task::PhysicalColumn>(ctx); }
}  // namespace legate::dataframe::argument
