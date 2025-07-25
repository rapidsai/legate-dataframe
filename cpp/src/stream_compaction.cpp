/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <legate.h>

#include <cudf/stream_compaction.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <stdexcept>

namespace legate::dataframe {
namespace task {

class ApplyBooleanMaskTask : public Task<ApplyBooleanMaskTask, OpCode::ApplyBooleanMask> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ApplyBooleanMask}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto tbl          = argument::get_next_input<PhysicalTable>(ctx);
    const auto boolean_mask = argument::get_next_input<PhysicalColumn>(ctx);
    auto output             = argument::get_next_output<PhysicalTable>(ctx);

    auto dummy_column_names = std::vector<std::string>(tbl.num_columns());
    std::iota(dummy_column_names.begin(), dummy_column_names.end(), 0);

    auto result =
      ARROW_RESULT(
        arrow::compute::CallFunction(
          "filter", {tbl.arrow_table_view(dummy_column_names), boolean_mask.arrow_array_view()}))
        .table();

    output.move_into(std::move(result));
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto tbl    = argument::get_next_input<PhysicalTable>(ctx);
    auto boolean_mask = argument::get_next_input<PhysicalColumn>(ctx);
    auto output       = argument::get_next_output<PhysicalTable>(ctx);

    auto ret = cudf::apply_boolean_mask(
      tbl.table_view(), boolean_mask.column_view(), ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
  }
};

}  // namespace task

LogicalTable apply_boolean_mask(const LogicalTable& tbl, const LogicalColumn& boolean_mask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto ret     = LogicalTable::empty_like(tbl);

  if (boolean_mask.cudf_type().id() != cudf::type_id::BOOL8) {
    throw std::invalid_argument("boolean mask column must have a bool dtype.");
  }

  legate::AutoTask task =
    runtime->create_task(get_library(), task::ApplyBooleanMaskTask::TASK_CONFIG.task_id());

  argument::add_next_input(task, tbl);
  argument::add_next_input(task, boolean_mask);
  argument::add_next_output(task, ret);

  runtime->submit(std::move(task));
  return ret;
}
}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::ApplyBooleanMaskTask::register_variants();
  return 0;
}();

}  // namespace
