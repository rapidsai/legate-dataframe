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

#include <legate.h>

#include <arrow/compute/api.h>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/unaryop.hpp>

namespace legate::dataframe {
namespace task {

/*static*/ void CastTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);

  auto cast = ARROW_RESULT(arrow::compute::Cast(
    input.arrow_array_view(), output.arrow_type(), arrow::compute::CastOptions::Unsafe()));
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(cast.make_array()));
  } else {
    output.move_into(std::move(cast.make_array()));
  }
}

/*static*/ void RoundTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto decimals    = argument::get_next_scalar<int32_t>(ctx);
  auto mode        = argument::get_next_scalar<std::string>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);

  arrow::compute::RoundOptions round_options;
  round_options.ndigits = decimals;
  if (mode == "half_away_from_zero") {
    round_options.round_mode = arrow::compute::RoundMode::HALF_TOWARDS_INFINITY;
  } else if (mode == "half_to_even") {
    round_options.round_mode = arrow::compute::RoundMode::HALF_TO_EVEN;
  } else {
    throw std::invalid_argument("Unsupported rounding method: " + mode);
  }

  auto res = ARROW_RESULT(arrow::compute::Round(input.arrow_array_view(), round_options));
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(res.make_array()));
  } else {
    output.move_into(std::move(res.make_array()));
  }
}

/*static*/ void UnaryOpTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  auto op          = argument::get_next_scalar<std::string>(ctx);
  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);
  auto result =
    ARROW_RESULT(arrow::compute::CallFunction(op, {input.arrow_array_view()})).make_array();
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(result));
  } else {
    output.move_into(std::move(result));
  }
}

}  // namespace task

LogicalColumn cast(const LogicalColumn& col, cudf::data_type to_type)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::CastTask::TASK_CONFIG.task_id());

  // Unary ops can return a scalar column for a scalar column input.
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = col.num_rows(); }
  auto ret     = LogicalColumn::empty_like(to_type, col.nullable(), col.is_scalar(), size);
  auto in_var  = argument::add_next_input(task, col);
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, in_var)); }
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn round(const LogicalColumn& col, int32_t digits, std::string mode)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::RoundTask::TASK_CONFIG.task_id());

  if (mode != "half_away_from_zero" && mode != "half_to_even") {
    throw std::invalid_argument("Unsupported rounding method: " + mode);
  }

  // Unary ops can return a scalar column for a scalar column input.
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = col.num_rows(); }
  auto ret    = LogicalColumn::empty_like(col.arrow_type(), col.nullable(), col.is_scalar(), size);
  auto in_var = argument::add_next_input(task, col);
  argument::add_next_scalar(task, digits);
  argument::add_next_scalar(task, mode);
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, in_var)); }
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn unary_operation(const LogicalColumn& col, std::string op)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::UnaryOpTask::TASK_CONFIG.task_id());

  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = col.num_rows(); }
  // Unary ops can return a scalar column for a scalar column input.
  auto ret = LogicalColumn::empty_like(col.cudf_type(), col.nullable(), col.is_scalar(), size);

  argument::add_next_scalar(task, op);
  argument::add_next_input(task, col);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::CastTask::register_variants();
  legate::dataframe::task::RoundTask::register_variants();
  legate::dataframe::task::UnaryOpTask::register_variants();
}

}  // namespace
