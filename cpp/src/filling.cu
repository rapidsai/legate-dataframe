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

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/filling.hpp>

namespace legate::dataframe::task {
/*static*/ void SequenceTask::gpu_variant(legate::TaskContext context)
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

  cudf::numeric_scalar<int64_t> cudf_init(local_init, true, ctx.stream(), ctx.mr());
  auto res = cudf::sequence(local_size, cudf_init, ctx.stream(), ctx.mr());

  output.move_into(std::move(res));
}
}  // namespace legate::dataframe::task
