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

#include <stdexcept>

#include <legate.h>

#include <cudf/replace.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/replace.hpp>

namespace legate::dataframe::task {
/*static*/ void ReplaceNullScalarTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto scalar_col  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);

  auto cudf_scalar = scalar_col.cudf_scalar();

  auto ret = cudf::replace_nulls(input.column_view(), *cudf_scalar, ctx.stream(), ctx.mr());

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
