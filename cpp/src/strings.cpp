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

#include <set>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/strings.hpp>

namespace legate::dataframe::task {

/*static*/ void StringMatchesTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);
  auto match_func  = argument::get_next_scalar<std::string>(ctx);
  auto pattern     = argument::get_next_scalar<std::string>(ctx);

  if (input.num_rows() <= 0) {
    if (!get_prefer_eager_allocations()) { output.bind_empty_data(); }
    return;
  }

  std::vector<arrow::Datum> args = {input.arrow_array_view()};
  auto options                   = arrow::compute::MatchSubstringOptions(pattern);

  auto datum_result = ARROW_RESULT(arrow::compute::CallFunction(match_func, args, &options));

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
  legate::dataframe::task::StringMatchesTask::register_variants();
}

}  // namespace

namespace legate::dataframe::strings {

LogicalColumn match(const std::string& match_func,
                    const LogicalColumn& column,
                    const std::string& pattern)
{
  auto runtime = legate::Runtime::get_runtime();

  if (column.arrow_type()->id() != arrow::Type::STRING &&
      column.arrow_type()->id() != arrow::Type::LARGE_STRING) {
    throw std::invalid_argument("matches requires string column");
  }

  static std::set<std::string> match_funcs(
    {"starts_with", "ends_with", "match_substring", "match_substring_regex"});
  if (match_funcs.count(match_func) == 0) {
    throw std::invalid_argument("Invalid match type: " + match_func);
  }

  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = column.num_rows(); }

  auto ret =
    LogicalColumn::empty_like(arrow::boolean(), column.nullable(), column.is_scalar(), size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::StringMatchesTask::TASK_CONFIG.task_id());

  auto col_var = argument::add_next_input(task, column);
  auto out_var = argument::add_next_output(task, ret);
  argument::add_next_scalar(task, match_func);
  argument::add_next_scalar(task, pattern);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, col_var)); }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe::strings
