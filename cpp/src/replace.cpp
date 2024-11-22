/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/scalar.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/replace.hpp>

namespace legate::dataframe {
namespace task {

class ReplaceNullScalarTask : public Task<ReplaceNullScalarTask, OpCode::ReplaceNullsWithScalar> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto scalar      = argument::get_next_scalar<ScalarArg>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);
    auto cudf_scalar = scalar.get_cudf(ctx.stream(), ctx.mr());

    auto ret = cudf::replace_nulls(input.column_view(), *cudf_scalar, ctx.stream(), ctx.mr());

    output.move_into(std::move(ret));
  }
};

}  // namespace task

LogicalColumn replace_nulls(const LogicalColumn& col, const ScalarArg& scalar)
{
  auto runtime = legate::Runtime::get_runtime();
  // Result needs to be nullable if the input is and the scalar is null also
  auto ret = LogicalColumn::empty_like(col.cudf_type(), col.nullable() && scalar.is_null());
  if (col.cudf_type() != scalar.cudf_type()) {
    throw std::invalid_argument("Scalar type does not match column type.");
  }
  legate::AutoTask task = runtime->create_task(get_library(), task::ReplaceNullScalarTask::TASK_ID);

  argument::add_next_input(task, col);
  argument::add_next_scalar(task, scalar);
  argument::add_next_output(task, ret);

  runtime->submit(std::move(task));
  return ret;
}
}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ReplaceNullScalarTask::register_variants();
}

}  // namespace