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

#include <cudf/types.hpp>
#include <legate.h>

#include <cudf/unary.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/unaryop.hpp>

namespace legate::dataframe {
namespace task {

class CastTask : public Task<CastTask, OpCode::Cast> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::Cast}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input                  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
    cudf::column_view col             = input.column_view();
    std::unique_ptr<cudf::column> ret = cudf::cast(col, output.cudf_type(), ctx.stream(), ctx.mr());
    output.move_into(std::move(ret), /* allow_copy */ true);
  }
};

class UnaryOpTask : public Task<UnaryOpTask, OpCode::UnaryOp> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::UnaryOp}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    auto op                           = argument::get_next_scalar<cudf::unary_operator>(ctx);
    const auto input                  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
    cudf::column_view col             = input.column_view();
    std::unique_ptr<cudf::column> ret = cudf::unary_operation(col, op, ctx.stream(), ctx.mr());
    output.move_into(std::move(ret), /* allow_copy */ true);
  }
};

}  // namespace task

LogicalColumn cast(const LogicalColumn& col, cudf::data_type to_type)
{
  if (!cudf::is_supported_cast(col.cudf_type(), to_type)) {
    throw std::invalid_argument("Cannot cast column to specified type");
  }

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

LogicalColumn unary_operation(const LogicalColumn& col, cudf::unary_operator op)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::UnaryOpTask::TASK_CONFIG.task_id());

  // Unary ops can return a scalar column for a scalar column input.
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = col.num_rows(); }
  auto ret = LogicalColumn::empty_like(col.cudf_type(), col.nullable(), col.is_scalar(), size);

  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::unary_operator>>(op));
  auto in_var  = argument::add_next_input(task, col);
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, in_var)); }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::CastTask::register_variants();
  legate::dataframe::task::UnaryOpTask::register_variants();
  return 0;
}();

}  // namespace
