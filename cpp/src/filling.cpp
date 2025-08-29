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

#include <legate.h>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/filling.hpp>

namespace legate::dataframe {
namespace task {

/*static*/ void SequenceTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  auto global_size = argument::get_next_scalar<size_t>(ctx);
  auto global_init = argument::get_next_scalar<int64_t>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);
  argument::get_parallel_launch_task(ctx);
  auto [local_start, local_size] = evenly_partition_work(global_size, ctx.rank, ctx.nranks);
  auto local_init                = global_init + local_start;

  if (local_size == 0) {
    output.bind_empty_data();
    return;
  }

  arrow::Int64Builder long_builder = arrow::Int64Builder();
  auto status                      = long_builder.Reserve(local_size);
  for (size_t i = 0; i < local_size; i++) {
    long_builder.UnsafeAppend(local_init + i);
  }
  auto local_array = ARROW_RESULT(long_builder.Finish());
  output.move_into(std::move(local_array));
}
}  // namespace task

LogicalColumn sequence(size_t size, int64_t init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto ret     = LogicalColumn::empty_like(arrow::int64(), false);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::SequenceTask::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, size);
  argument::add_next_scalar(task, init);
  argument::add_next_output(task, ret);
  argument::add_parallel_launch_task(task);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::SequenceTask::register_variants();
}

}  // namespace
