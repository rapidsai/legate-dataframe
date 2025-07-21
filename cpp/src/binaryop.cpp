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

#include <arrow/compute/api.h>
#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

cudf::binary_operator arrow_to_cudf_binary_op(std::string op, legate::Type output_type)
{
  // Arrow binary operators taken from the below list,
  // where an equivalent cudf binary operator exists.
  // https://arrow.apache.org/docs/cpp/compute.html#element-wise-scalar-functions
  // https://docs.rapids.ai/api/libcudf/stable/group__transformation__binaryops
  std::unordered_map<std::string, cudf::binary_operator> arrow_to_cudf_ops = {
    {"add", cudf::binary_operator::ADD},
    {"divide", cudf::binary_operator::DIV},
    {"multiply", cudf::binary_operator::MUL},
    {"power", cudf::binary_operator::POW},
    {"subtract", cudf::binary_operator::SUB},
    {"bit_wise_and", cudf::binary_operator::BITWISE_AND},
    {"bit_wise_or", cudf::binary_operator::BITWISE_OR},
    {"bit_wise_xor", cudf::binary_operator::BITWISE_XOR},
    {"shift_left", cudf::binary_operator::SHIFT_LEFT},
    {"shift_right", cudf::binary_operator::SHIFT_RIGHT},
    {"logb", cudf::binary_operator::LOG_BASE},
    {"atan2", cudf::binary_operator::ATAN2},
    {"equal", cudf::binary_operator::EQUAL},
    {"greater", cudf::binary_operator::GREATER},
    {"greater_equal", cudf::binary_operator::GREATER_EQUAL},
    {"less", cudf::binary_operator::LESS},
    {"less_equal", cudf::binary_operator::LESS_EQUAL},
    {"not_equal", cudf::binary_operator::NOT_EQUAL},
    // logical operators:
    {"and", cudf::binary_operator::LOGICAL_AND},
    {"or", cudf::binary_operator::LOGICAL_OR},
    {"and_kleene", cudf::binary_operator::NULL_LOGICAL_AND},
    {"or_kleene", cudf::binary_operator::NULL_LOGICAL_OR},
  };

  // Cudf has a special case for powers with integers
  // https://github.com/rapidsai/cudf/issues/10178#issuecomment-3004143727
  if (op == "power" && output_type.to_string().find("int") != std::string::npos) {
    return cudf::binary_operator::INT_POW;
  }

  if (arrow_to_cudf_ops.find(op) != arrow_to_cudf_ops.end()) { return arrow_to_cudf_ops[op]; }
  throw std::invalid_argument("Could not find cudf binary operator matching: " + op);
  return cudf::binary_operator::INVALID_BINARY;
}

class BinaryOpColColTask : public Task<BinaryOpColColTask, OpCode::BinaryOpColCol> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::BinaryOpColCol}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    auto op        = argument::get_next_scalar<std::string>(ctx);
    const auto lhs = argument::get_next_input<PhysicalColumn>(ctx);
    const auto rhs = argument::get_next_input<PhysicalColumn>(ctx);
    auto output    = argument::get_next_output<PhysicalColumn>(ctx);

    std::vector<arrow::Datum> args(2);
    if (lhs.num_rows() == 1) {
      auto scalar = ARROW_RESULT(lhs.arrow_array_view()->GetScalar(0));
      args[0]     = scalar;
    } else {
      args[0] = lhs.arrow_array_view();
    }
    if (rhs.num_rows() == 1) {
      auto scalar = ARROW_RESULT(rhs.arrow_array_view()->GetScalar(0));
      args[1]     = scalar;
    } else {
      args[1] = rhs.arrow_array_view();
    }

    if (output.cudf_type().id() == cudf::type_id::BOOL8 &&
        (op == "and" || op == "or" || op == "and_kleene" || op == "or_kleene")) {
      // arrow doesn't seem to cast for the user for logical ops.
      args[0] = ARROW_RESULT(arrow::compute::Cast(args[0], arrow::boolean()));
      args[1] = ARROW_RESULT(arrow::compute::Cast(args[1], arrow::boolean()));
    }

    // Result may be scalar or array
    auto datum_result = ARROW_RESULT(arrow::compute::CallFunction(op, args));

    // Coerce the output type if necessary
    auto arrow_result_type = to_arrow_type(output.cudf_type().id());
    if (datum_result.type() != arrow_result_type) {
      auto coerced_result = ARROW_RESULT(arrow::compute::Cast(
        datum_result, arrow_result_type, arrow::compute::CastOptions::Unsafe()));
      datum_result        = std::move(coerced_result);
    }

    if (datum_result.is_scalar()) {
      auto as_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*datum_result.scalar(), 1));
      output.move_into(std::move(as_array));
    } else {
      output.move_into(std::move(datum_result.make_array()));
    }
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    auto arrow_op  = argument::get_next_scalar<std::string>(ctx);
    const auto lhs = argument::get_next_input<PhysicalColumn>(ctx);
    const auto rhs = argument::get_next_input<PhysicalColumn>(ctx);
    auto output    = argument::get_next_output<PhysicalColumn>(ctx);
    auto op        = arrow_to_cudf_binary_op(arrow_op, output.type());

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
        lhs.column_view(), rhs.column_view(), op, output.cudf_type(), ctx.stream(), ctx.mr());
    }
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(ret));
    } else {
      output.move_into(std::move(ret));
    }
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
                               std::string op,
                               cudf::data_type output_type)
{
  auto runtime = legate::Runtime::get_runtime();

  // Check if the op is valid before we enter the task
  // This allows us to to throw nicely
  if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) > 0) {
    // Throws if op doesn't exist
    task::arrow_to_cudf_binary_op(op, to_legate_type(output_type.id()));
  } else {
    auto result = arrow::compute::GetFunctionRegistry()->GetFunction(op);
    if (!result.ok()) {
      throw std::invalid_argument("Could not find arrow binary operator matching: " + op);
    }
  }

  bool nullable      = lhs.nullable() || rhs.nullable();
  auto scalar_result = lhs.is_scalar() && rhs.is_scalar();
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) { size = lhs.is_scalar() ? rhs.num_rows() : lhs.num_rows(); }
  auto ret = LogicalColumn::empty_like(std::move(output_type), nullable, scalar_result, size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::BinaryOpColColTask::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, op);

  /* Add the inputs, broadcast if scalar.  If both aren't scalar align them */
  auto lhs_var = argument::add_next_input(task, lhs, /* broadcast */ lhs.is_scalar());
  auto rhs_var = argument::add_next_input(task, rhs, /* broadcast */ rhs.is_scalar());
  if (!rhs.is_scalar() && !lhs.is_scalar()) {
    task.add_constraint(legate::align(lhs_var, rhs_var));
  }
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) {
    task.add_constraint(legate::align(out_var, lhs.is_scalar() ? rhs_var : lhs_var));
  }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
