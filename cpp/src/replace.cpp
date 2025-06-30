/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <stdexcept>

#include <legate.h>

#include <cudf/replace.hpp>

#include <arrow/compute/api.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/replace.hpp>

namespace legate::dataframe {
namespace task {

class ReplaceNullScalarTask : public Task<ReplaceNullScalarTask, OpCode::ReplaceNullsWithScalar> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ReplaceNullsWithScalar}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto scalar_col  = argument::get_next_input<PhysicalColumn>(ctx);

    auto arrow_input = input.arrow_array_view();

    auto scalar = ARROW_RESULT(scalar_col.arrow_array_view()->GetScalar(0));
    auto output = argument::get_next_output<PhysicalColumn>(ctx);

    // Cast the scalar to the input type if necessary
    if (*scalar->type != *arrow_input->type()) {
      scalar = ARROW_RESULT(scalar->CastTo(input.arrow_array_view()->type()));
    }
    auto datum_result =
      ARROW_RESULT(arrow::compute::CallFunction("coalesce", {arrow_input, scalar}));
    output.move_into(std::move(datum_result.make_array()));
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto scalar_col  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);

    auto cudf_scalar = scalar_col.cudf_scalar();

    auto ret = cudf::replace_nulls(input.column_view(), *cudf_scalar, ctx.stream(), ctx.mr());

    output.move_into(std::move(ret));
  }
};

}  // namespace task

LogicalColumn replace_nulls(const LogicalColumn& col, const LogicalColumn& scalar)
{
  auto runtime = legate::Runtime::get_runtime();
  // Result needs to be nullable if the input is and the scalar is also.
  // NOTE: We possibly should bite the bullet here and check if the scalar is null
  // or not.  That is blocking, though.
  auto ret = LogicalColumn::empty_like(col.cudf_type(), col.nullable() && scalar.nullable());
  if (col.cudf_type() != scalar.cudf_type()) {
    throw std::invalid_argument("Scalar type does not match column type.");
  }
  if (!scalar.is_scalar()) {
    // NOTE: We could be graceful here and check if it has 1 rows.
    throw std::invalid_argument("Scalar column must be marked as scalar.");
  }
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ReplaceNullScalarTask::TASK_CONFIG.task_id());

  argument::add_next_input(task, col);
  argument::add_next_input(task, scalar, /* broadcast */ true);
  argument::add_next_output(task, ret);

  runtime->submit(std::move(task));
  return ret;
}
}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::ReplaceNullScalarTask::register_variants();
  return 0;
}();

}  // namespace
