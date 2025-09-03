/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/search.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/search.hpp>

namespace legate::dataframe::task {

/*static*/ void ContainsTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto haystack = argument::get_next_input<PhysicalColumn>(ctx);
  const auto needles  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output         = argument::get_next_output<PhysicalColumn>(ctx);

  auto ret = cudf::contains(haystack.column_view(), needles.column_view(), ctx.stream(), ctx.mr());

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
