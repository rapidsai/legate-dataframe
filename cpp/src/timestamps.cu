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

#include <string>

#include <cudf/datetime.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/unary.hpp>

#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/timestamps.hpp>

namespace legate::dataframe::task {

/*static*/ void ToTimestampsTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto format = argument::get_next_scalar<std::string>(ctx);
  const auto input  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output       = argument::get_next_output<PhysicalColumn>(ctx);

  std::unique_ptr<cudf::column> ret = cudf::strings::to_timestamps(
    input.column_view(), output.cudf_type(), format, ctx.stream(), ctx.mr());
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

cudf::datetime::datetime_component to_cudf_component(std::string component)
{
  static const std::map<std::string, cudf::datetime::datetime_component> components = {
    {"year", cudf::datetime::datetime_component::YEAR},
    {"month", cudf::datetime::datetime_component::MONTH},
    {"day", cudf::datetime::datetime_component::DAY},
    {"day_of_week", cudf::datetime::datetime_component::WEEKDAY},
    {"hour", cudf::datetime::datetime_component::HOUR},
    {"minute", cudf::datetime::datetime_component::MINUTE},
    {"second", cudf::datetime::datetime_component::SECOND},
    {"millisecond", cudf::datetime::datetime_component::MILLISECOND},
    {"microsecond", cudf::datetime::datetime_component::MICROSECOND},
    {"nanosecond", cudf::datetime::datetime_component::NANOSECOND}};
  auto it = components.find(component);
  if (it == components.end()) {
    throw std::invalid_argument("Cudf does not support datetime component: " + component);
  }
  return it->second;
}

/*static*/ void ExtractTimestampComponentTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto component = argument::get_next_scalar<std::string>(ctx);
  const auto input     = argument::get_next_input<PhysicalColumn>(ctx);
  auto output          = argument::get_next_output<PhysicalColumn>(ctx);

  std::unique_ptr<cudf::column> ret;
  ret = cudf::datetime::extract_datetime_component(
    input.column_view(), to_cudf_component(component), ctx.stream(), ctx.mr());

  // Cast to int64
  ret = cudf::cast(ret->view(), cudf::data_type(cudf::type_id::INT64), ctx.stream(), ctx.mr());

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
