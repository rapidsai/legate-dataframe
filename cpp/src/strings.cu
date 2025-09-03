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

#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/types.hpp>
#include <legate.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/strings.hpp>

namespace legate::dataframe::task {

/*static*/ void StringMatchesTask::gpu_variant(legate::TaskContext context)
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

  std::unique_ptr<cudf::column> ret;
  auto cudf_pattern = cudf::string_scalar(pattern);

  if (match_func == "starts_with") {
    ret = cudf::strings::starts_with(input.column_view(), cudf_pattern, ctx.stream(), ctx.mr());
  } else if (match_func == "ends_with") {
    ret = cudf::strings::ends_with(input.column_view(), cudf_pattern, ctx.stream(), ctx.mr());
  } else if (match_func == "match_substring") {
    ret = cudf::strings::contains(input.column_view(), cudf_pattern, ctx.stream(), ctx.mr());
  } else if (match_func == "match_substring_regex") {
    auto prog = cudf::strings::regex_program::create(pattern);
    ret       = cudf::strings::contains_re(input.column_view(), *prog, ctx.stream(), ctx.mr());
  } else {
    throw std::runtime_error("Invalid match type in gpu task.");
  }

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
