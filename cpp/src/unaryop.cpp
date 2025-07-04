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

#include <arrow/compute/api.h>
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

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);

    auto cast = ARROW_RESULT(arrow::compute::Cast(
      input.arrow_array_view(), output.arrow_type(), arrow::compute::CastOptions::Unsafe()));
    output.move_into(std::move(cast.make_array()));
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input                  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
    cudf::column_view col             = input.column_view();
    std::unique_ptr<cudf::column> ret = cudf::cast(col, output.cudf_type(), ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
  }
};

cudf::unary_operator arrow_to_cudf_unary_op(std::string op)
{
  // Arrow unary operators taken from the below list,
  // where an equivalent cudf unary operator exists.
  // https://arrow.apache.org/docs/cpp/compute.html#element-wise-scalar-functions
  // https://docs.rapids.ai/api/libcudf/stable/group__transformation__unaryops
  std::unordered_map<std::string, cudf::unary_operator> arrow_to_cudf_ops = {
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

  if (arrow_to_cudf_ops.find(op) != arrow_to_cudf_ops.end()) { return arrow_to_cudf_ops[op]; }
  throw std::invalid_argument("Could not find cudf binary operator matching: " + op);
  return cudf::unary_operator::ABS;
}

class UnaryOpTask : public Task<UnaryOpTask, OpCode::UnaryOp> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::UnaryOp}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    auto op          = argument::get_next_scalar<std::string>(ctx);
    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);
    auto result =
      ARROW_RESULT(arrow::compute::CallFunction(op, {input.arrow_array_view()})).make_array();
    output.move_into(std::move(result));
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    auto op               = argument::get_next_scalar<std::string>(ctx);
    const auto input      = argument::get_next_input<PhysicalColumn>(ctx);
    auto output           = argument::get_next_output<PhysicalColumn>(ctx);
    cudf::column_view col = input.column_view();
    std::unique_ptr<cudf::column> ret =
      cudf::unary_operation(col, arrow_to_cudf_unary_op(op), ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
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
  auto ret = LogicalColumn::empty_like(to_type, col.nullable(), col.is_scalar());
  argument::add_next_input(task, col);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn unary_operation(const LogicalColumn& col, std::string op)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::UnaryOpTask::TASK_CONFIG.task_id());

  // Unary ops can return a scalar column for a scalar column input.
  auto ret = LogicalColumn::empty_like(col.cudf_type(), col.nullable(), col.is_scalar());
  argument::add_next_scalar(task, op);
  argument::add_next_input(task, col);
  argument::add_next_output(task, ret);
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
