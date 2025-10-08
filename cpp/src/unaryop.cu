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

#include <cudf/round.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <legate.h>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/unaryop.hpp>

namespace legate::dataframe::task {

/*static*/ void CastTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto input                  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
  cudf::column_view col             = input.column_view();
  std::unique_ptr<cudf::column> ret = cudf::cast(col, output.cudf_type(), ctx.stream(), ctx.mr());
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

/*static*/ void RoundTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto input      = argument::get_next_input<PhysicalColumn>(ctx);
  auto decimal_places   = argument::get_next_scalar<int32_t>(ctx);
  auto mode             = argument::get_next_scalar<std::string>(ctx);
  auto output           = argument::get_next_output<PhysicalColumn>(ctx);
  cudf::column_view col = input.column_view();
  cudf::rounding_method rounding_method;
  if (mode == "half_away_from_zero") {
    rounding_method = cudf::rounding_method::HALF_UP;
  } else if (mode == "half_to_even") {
    rounding_method = cudf::rounding_method::HALF_EVEN;
  } else {
    throw std::invalid_argument("Unsupported rounding method: " + mode);
  }
  // TODO(seberg): Need to switch to round_decimal, but it failed tests due to
  // some input types in our tests and I have not yet checked why or what to use.
  std::unique_ptr<cudf::column> ret =
    cudf::round(col, decimal_places, rounding_method, ctx.stream(), ctx.mr());
  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

/*static*/ void UnaryOpTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  auto op               = argument::get_next_scalar<std::string>(ctx);
  const auto input      = argument::get_next_input<PhysicalColumn>(ctx);
  auto output           = argument::get_next_output<PhysicalColumn>(ctx);
  cudf::column_view col = input.column_view();

  // Arrow unary operators taken from the below list,
  // where an equivalent cudf unary operator exists.
  // https://arrow.apache.org/docs/cpp/compute.html#element-wise-scalar-functions
  // https://docs.rapids.ai/api/libcudf/stable/group__transformation__unaryops
  static const std::unordered_map<std::string, cudf::unary_operator> arrow_to_cudf_ops = {
    {"sin", cudf::unary_operator::SIN},       {"cos", cudf::unary_operator::COS},
    {"tan", cudf::unary_operator::TAN},       {"asin", cudf::unary_operator::ARCSIN},
    {"acos", cudf::unary_operator::ARCCOS},   {"atan", cudf::unary_operator::ARCTAN},
    {"sinh", cudf::unary_operator::SINH},     {"cosh", cudf::unary_operator::COSH},
    {"tanh", cudf::unary_operator::TANH},     {"asinh", cudf::unary_operator::ARCSINH},
    {"acosh", cudf::unary_operator::ARCCOSH}, {"atanh", cudf::unary_operator::ARCTANH},
    {"exp", cudf::unary_operator::EXP},       {"ln", cudf::unary_operator::LOG},
    {"sqrt", cudf::unary_operator::SQRT},     {"ceil", cudf::unary_operator::CEIL},
    {"floor", cudf::unary_operator::FLOOR},   {"abs", cudf::unary_operator::ABS},
    {"round", cudf::unary_operator::RINT},    {"bit_wise_not", cudf::unary_operator::BIT_INVERT},
    {"invert", cudf::unary_operator::NOT},    {"negate", cudf::unary_operator::NEGATE}};

  std::unique_ptr<cudf::column> ret;
  auto it = arrow_to_cudf_ops.find(op);
  if (it != arrow_to_cudf_ops.end()) {
    ret = cudf::unary_operation(col, it->second, ctx.stream(), ctx.mr());
  } else if (op == "is_nan") {
    ret = cudf::is_nan(col, ctx.stream(), ctx.mr());
    // As of 25.06 does not propagate nulls (historic reasons with pandas likely)
    if (col.has_nulls()) {
      auto null_mask = cudf::copy_bitmask(col, ctx.stream(), ctx.mr());
      ret->set_null_mask(std::move(null_mask), col.null_count());
    }
  } else if (op == "is_null") {
    ret = cudf::is_null(col, ctx.stream(), ctx.mr());
  } else if (op == "is_valid") {
    ret = cudf::is_valid(col, ctx.stream(), ctx.mr());
  } else {
    throw std::invalid_argument("Could not find cudf binary operator matching: " + op);
  }

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
