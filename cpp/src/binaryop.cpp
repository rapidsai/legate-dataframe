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

#include <arrow/compute/api.h>
#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

/*static*/ void BinaryOpColColTask::cpu_variant(legate::TaskContext context)
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

  if (output.arrow_type() == arrow::boolean() &&
      (op == "and" || op == "or" || op == "and_kleene" || op == "or_kleene")) {
    // arrow doesn't seem to cast for the user for logical ops.
    args[0] = ARROW_RESULT(arrow::compute::Cast(args[0], arrow::boolean()));
    args[1] = ARROW_RESULT(arrow::compute::Cast(args[1], arrow::boolean()));
  }

  /*
  std::cout << lhs.arrow_array_view()->type()->ToString() << std::endl;
  std::cout << rhs.arrow_array_view()->type()->ToString() << std::endl;
  std::cout << lhs.arrow_array_view()->ToString() << std::endl;
  std::cout << rhs.arrow_array_view()->ToString() << std::endl;
  */

  // Result may be scalar or array
  auto datum_result = ARROW_RESULT(arrow::compute::CallFunction(op, args));

  // Coerce the output type if necessary
  if (datum_result.type() != output.arrow_type()) {
    auto coerced_result = ARROW_RESULT(arrow::compute::Cast(
      datum_result, output.arrow_type(), arrow::compute::CastOptions::Unsafe()));
    datum_result        = std::move(coerced_result);
  }

  if (datum_result.is_scalar()) {
    auto as_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*datum_result.scalar(), 1));
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(as_array));
    } else {
      output.move_into(std::move(as_array));
    }
  } else {
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(datum_result.make_array()));
    } else {
      output.move_into(std::move(datum_result.make_array()));
    }
  }
}

}  // namespace legate::dataframe::task

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::BinaryOpColColTask::register_variants();
}

}  // namespace

namespace legate::dataframe {

LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const LogicalColumn& rhs,
                               std::string op,
                               std::shared_ptr<arrow::DataType> output_type)
{
  auto runtime = legate::Runtime::get_runtime();

  // Check if the op is valid before we enter the task
  // This allows us to to throw nicely
  if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) > 0) {
    if (task::cudf_supported_binary_ops.count(op) == 0) {
      throw std::invalid_argument("Unsupported binary operator: " + op);
    }
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
  auto ret = LogicalColumn::empty_like(output_type, nullable, scalar_result, size);
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
