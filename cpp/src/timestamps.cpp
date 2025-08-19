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

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <cudf/datetime.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/unary.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/timestamps.hpp>

namespace legate::dataframe {
namespace task {

class ToTimestampsTask : public Task<ToTimestampsTask, OpCode::ToTimestamps> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto format   = argument::get_next_scalar<std::string>(ctx);
    const auto input    = argument::get_next_input<PhysicalColumn>(ctx);
    auto output         = argument::get_next_output<PhysicalColumn>(ctx);
    auto timestamp_type = std::dynamic_pointer_cast<arrow::TimestampType>(output.arrow_type());
    if (!timestamp_type) { throw std::invalid_argument("Output type must be a timestamp type"); }
    arrow::compute::StrptimeOptions options(format, timestamp_type->unit());
    auto result =
      ARROW_RESULT(arrow::compute::CallFunction("strptime", {input.arrow_array_view()}, &options))
        .make_array();
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(result));
    } else {
      output.move_into(std::move(result));
    }
  }

  static void gpu_variant(legate::TaskContext context)
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
};

cudf::datetime::datetime_component to_cudf_component(std::string component)
{
  std::map<std::string, cudf::datetime::datetime_component> components = {
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

class ExtractTimestampComponentTask
  : public Task<ExtractTimestampComponentTask, OpCode::ExtractTimestampComponent> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto component = argument::get_next_scalar<std::string>(ctx);
    const auto input     = argument::get_next_input<PhysicalColumn>(ctx);
    auto output          = argument::get_next_output<PhysicalColumn>(ctx);

    std::shared_ptr<arrow::Array> result;
    if (component == "day_of_week") {
      arrow::compute::DayOfWeekOptions options(false);  // Count from zero false to match cudf
      result = ARROW_RESULT(
                 arrow::compute::CallFunction("day_of_week", {input.arrow_array_view()}, &options))
                 .make_array();
    } else {
      result = ARROW_RESULT(arrow::compute::CallFunction(component, {input.arrow_array_view()}))
                 .make_array();
    }

    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(result));
    } else {
      output.move_into(std::move(result));
    }
  }
  static void gpu_variant(legate::TaskContext context)
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
};

}  // namespace task

LogicalColumn to_timestamps(const LogicalColumn& input,
                            std::shared_ptr<arrow::DataType> timestamp_type,
                            std::string format)
{
  auto runtime = legate::Runtime::get_runtime();
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = input.num_rows(); }
  auto ret = LogicalColumn::empty_like(timestamp_type, input.nullable(), false, size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ToTimestampsTask::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, std::move(format));
  auto in_var  = argument::add_next_input(task, input);
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, in_var)); }
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn extract_timestamp_component(const LogicalColumn& input, std::string component)
{
  auto timestamp_type = std::dynamic_pointer_cast<arrow::TimestampType>(input.arrow_type());
  if (!timestamp_type) {
    throw std::invalid_argument("extract_timestamp_component() input must be timestamp");
  }

  auto runtime = legate::Runtime::get_runtime();
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = input.num_rows(); }
  auto ret = LogicalColumn::empty_like(arrow::int64(), input.nullable(), false, size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ExtractTimestampComponentTask::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, component);
  auto in_var  = argument::add_next_input(task, input);
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, in_var)); }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ToTimestampsTask::register_variants();
  legate::dataframe::task::ExtractTimestampComponentTask::register_variants();
}

}  // namespace
