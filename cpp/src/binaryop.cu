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

cudf::binary_operator arrow_to_cudf_binary_op(std::string op, legate::Type output_type)
{
  // Arrow binary operators taken from the below list,
  // where an equivalent cudf binary operator exists.
  // https://arrow.apache.org/docs/cpp/compute.html#element-wise-scalar-functions
  // https://docs.rapids.ai/api/libcudf/stable/group__transformation__binaryops
  std::unordered_map<std::string, cudf::binary_operator> arrow_to_cudf_ops = {
    {"add", cudf::binary_operator::ADD},
    // NOTE: if we enable true divide here, should improve polars side.
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

/*static*/ void BinaryOpColColTask::gpu_variant(legate::TaskContext context)
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

}  // namespace legate::dataframe::task
