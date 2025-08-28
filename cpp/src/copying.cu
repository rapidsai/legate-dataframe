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

#include <cudf/copying.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/copying.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

/*static*/ void CopyIfElseTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto cond = argument::get_next_input<PhysicalColumn>(ctx);
  const auto lhs  = argument::get_next_input<PhysicalColumn>(ctx);
  const auto rhs  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output     = argument::get_next_output<PhysicalColumn>(ctx);

  std::unique_ptr<cudf::column> ret;
  /*
   * Use scalars if inputs are to ensure broadcasting works, cond is always a column.
   * This unfortunately requires 4 cases (all 4 overloads provided by libcudf).
   */
  if (lhs.num_rows() == 1 && rhs.num_rows() != 1) {
    auto lhs_scalar = lhs.cudf_scalar();
    ret             = cudf::copy_if_else(
      *lhs_scalar, rhs.column_view(), cond.column_view(), ctx.stream(), ctx.mr());
  } else if (rhs.num_rows() == 1 && lhs.num_rows() != 1) {
    auto rhs_scalar = rhs.cudf_scalar();
    ret             = cudf::copy_if_else(
      lhs.column_view(), *rhs_scalar, cond.column_view(), ctx.stream(), ctx.mr());
  } else if (lhs.num_rows() == 1 && rhs.num_rows() == 1) {
    auto lhs_scalar = lhs.cudf_scalar();
    auto rhs_scalar = rhs.cudf_scalar();
    ret = cudf::copy_if_else(*lhs_scalar, *rhs_scalar, cond.column_view(), ctx.stream(), ctx.mr());
  } else {
    ret = cudf::copy_if_else(
      lhs.column_view(), rhs.column_view(), cond.column_view(), ctx.stream(), ctx.mr());
  }

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

}  // namespace legate::dataframe::task
