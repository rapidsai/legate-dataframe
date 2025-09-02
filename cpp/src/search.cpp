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

#include <arrow/compute/api.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/search.hpp>

namespace legate::dataframe::task {

/*static*/ void ContainsTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto haystack = argument::get_next_input<PhysicalColumn>(ctx);
  const auto needles  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output         = argument::get_next_output<PhysicalColumn>(ctx);

  // "is_in" has swapped argument order compared to "contains"
  std::vector<arrow::Datum> args{{needles.arrow_array_view()}};
  auto options = arrow::compute::SetLookupOptions(
    haystack.arrow_array_view(), arrow::compute::SetLookupOptions::NullMatchingBehavior::EMIT_NULL);
  auto datum_result = ARROW_RESULT(arrow::compute::CallFunction("is_in", args, &options));

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(datum_result.make_array()));
  } else {
    output.move_into(std::move(datum_result.make_array()));
  }
}

}  // namespace legate::dataframe::task

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ContainsTask::register_variants();
}

}  // namespace

namespace legate::dataframe {

LogicalColumn contains(const LogicalColumn& haystack, const LogicalColumn& needles)
{
  auto runtime = legate::Runtime::get_runtime();

  // Check if types are compatible
  if (*haystack.arrow_type() != *needles.arrow_type()) {
    throw std::invalid_argument("Haystack and needles must have the same type");
  }

  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = needles.num_rows(); }

  // Result is always boolean
  auto ret =
    LogicalColumn::empty_like(arrow::boolean(), needles.nullable(), needles.is_scalar(), size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ContainsTask::TASK_CONFIG.task_id());

  /* Add the inputs, broadcast if scalar. If both aren't scalar align them */
  auto haystack_var = argument::add_next_input(task, haystack, /* broadcast */ true);
  auto needles_var  = argument::add_next_input(task, needles);
  auto out_var      = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, needles_var)); }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
