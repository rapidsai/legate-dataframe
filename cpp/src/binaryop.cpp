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

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

class BinaryOpColColTask : public Task<BinaryOpColColTask, OpCode::BinaryOpColCol> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::BinaryOpColCol}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    auto op        = argument::get_next_scalar<cudf::binary_operator>(ctx);
    const auto lhs = argument::get_next_input<PhysicalColumn>(ctx);
    const auto rhs = argument::get_next_input<PhysicalColumn>(ctx);
    auto output    = argument::get_next_output<PhysicalColumn>(ctx);

    std::unique_ptr<cudf::column> ret;
    /*
     * If one (not both) are length 1, use scalars as cudf doesn't allow
     * broadcast binary operations.
     */
    if (lhs.num_rows() == 1 && rhs.num_rows() != 1) {
      auto lhs_scalar = lhs.cudf_scalar();
      ret             = cudf::binary_operation(
        *lhs_scalar, rhs.column_view(), op, output.cudf_type(), ctx.stream(), ctx.mr());
    } else if (rhs.num_rows() == 1 && lhs.num_rows() != 1) {
      auto rhs_scalar = rhs.cudf_scalar();
      ret             = cudf::binary_operation(
        lhs.column_view(), *rhs_scalar, op, output.cudf_type(), ctx.stream(), ctx.mr());
    } else {
      ret = cudf::binary_operation(
        lhs.column_view(), rhs.column_view(), op, output.cudf_type(), context.get_task_stream());
    }
    output.move_into(std::move(ret));
  }
};

}  // namespace legate::dataframe::task

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::BinaryOpColColTask::register_variants();
  return 0;
}();

}  // namespace

namespace legate::dataframe {

LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const LogicalColumn& rhs,
                               cudf::binary_operator op,
                               cudf::data_type output_type)
{
  auto runtime  = legate::Runtime::get_runtime();
  bool nullable = lhs.nullable() || rhs.nullable();

  auto scalar_result = lhs.is_scalar() && rhs.is_scalar();
  auto ret           = LogicalColumn::empty_like(std::move(output_type), nullable, scalar_result);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::BinaryOpColColTask::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::binary_operator>>(op));

  /* Add the inputs, broadcast if scalar.  If both aren't scalar align them */
  auto lhs_var = argument::add_next_input(task, lhs, /* broadcast */ lhs.is_scalar());
  auto rhs_var = argument::add_next_input(task, rhs, /* broadcast */ rhs.is_scalar());
  if (!rhs.is_scalar() && !lhs.is_scalar()) {
    task.add_constraint(legate::align(lhs_var, rhs_var));
  }
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
