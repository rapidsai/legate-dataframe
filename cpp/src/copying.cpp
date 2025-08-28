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
#include <legate_dataframe/copying.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

/*static*/ void CopyIfElseTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto cond = argument::get_next_input<PhysicalColumn>(ctx);
  const auto lhs  = argument::get_next_input<PhysicalColumn>(ctx);
  const auto rhs  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output     = argument::get_next_output<PhysicalColumn>(ctx);

  std::vector<arrow::Datum> args;
  args.reserve(3);
  for (auto& column : {&cond, &lhs, &rhs}) {
    if (column->num_rows() == 1) {
      auto scalar = ARROW_RESULT(column->arrow_array_view()->GetScalar(0));
      args.emplace_back(scalar);
    } else {
      args.emplace_back(column->arrow_array_view());
    }
  }

  // Result may be scalar or array
  auto datum_result = ARROW_RESULT(arrow::compute::CallFunction("if_else", args));

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
  legate::dataframe::task::CopyIfElseTask::register_variants();
}

}  // namespace

namespace legate::dataframe {

LogicalColumn copy_if_else(const LogicalColumn& cond,
                           const LogicalColumn& lhs,
                           const LogicalColumn& rhs)
{
  if (*lhs.arrow_type() != *rhs.arrow_type()) {
    throw std::invalid_argument("lhs and rhs must have the same type");
  }
  if (*cond.arrow_type() != *arrow::boolean()) {
    throw std::invalid_argument("cond must be a boolean column");
  }

  auto runtime = legate::Runtime::get_runtime();

  bool nullable      = lhs.nullable() || rhs.nullable();
  auto scalar_result = lhs.is_scalar() && rhs.is_scalar() && cond.is_scalar();
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) {
    size = std::max(lhs.num_rows(), std::max(rhs.num_rows(), cond.num_rows()));
  }
  auto ret = LogicalColumn::empty_like(lhs.arrow_type(), nullable, scalar_result, size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::CopyIfElseTask::TASK_CONFIG.task_id());

  /* Add the inputs, broadcast if scalar.  If both aren't scalar align them */
  auto cond_var = argument::add_next_input(task, cond, /* broadcast */ cond.is_scalar());
  auto lhs_var  = argument::add_next_input(task, lhs, /* broadcast */ lhs.is_scalar());
  auto rhs_var  = argument::add_next_input(task, rhs, /* broadcast */ rhs.is_scalar());
  if (!lhs.is_scalar()) { task.add_constraint(legate::align(lhs_var, cond_var)); }
  if (!rhs.is_scalar()) { task.add_constraint(legate::align(rhs_var, cond_var)); }
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, cond_var)); }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
