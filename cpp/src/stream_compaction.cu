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

#include <legate.h>

#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/stream_compaction.hpp>
#include <stdexcept>

namespace legate::dataframe::task {

/*static*/ void ApplyBooleanMaskTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto tbl    = argument::get_next_input<PhysicalTable>(ctx);
  auto boolean_mask = argument::get_next_input<PhysicalColumn>(ctx);
  auto output       = argument::get_next_output<PhysicalTable>(ctx);

  auto ret =
    cudf::apply_boolean_mask(tbl.table_view(), boolean_mask.column_view(), ctx.stream(), ctx.mr());
  output.move_into(std::move(ret));
}

/*static*/ void DistinctTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  auto table            = argument::get_next_input<PhysicalTable>(ctx);
  auto output           = argument::get_next_output<PhysicalTable>(ctx);
  auto key_col_idx      = argument::get_next_scalar_vector<int32_t>(ctx);
  auto high_cardinality = argument::get_next_scalar<bool>(ctx);

  std::unique_ptr<cudf::table> working_table;
  auto working_view = table.table_view();

  if (!high_cardinality && table.is_partitioned()) {
    // If the table has few unique entries, we can reduce work by
    // calling distinct before repartitioning.
    working_table = cudf::distinct(table.table_view(),
                                   key_col_idx,
                                   cudf::duplicate_keep_option::KEEP_ANY,
                                   cudf::null_equality::EQUAL,
                                   cudf::nan_equality::ALL_EQUAL,
                                   ctx.stream(),
                                   ctx.mr());
    working_view  = working_table->view();
  }
  working_table = repartition_by_hash(ctx, working_view, key_col_idx);
  working_view  = working_table->view();

  auto ret = cudf::distinct(working_view,
                            key_col_idx,
                            cudf::duplicate_keep_option::KEEP_ANY,
                            cudf::null_equality::EQUAL,
                            cudf::nan_equality::ALL_EQUAL,
                            ctx.stream(),
                            ctx.mr());
  output.move_into(std::move(ret));
}

}  // namespace legate::dataframe::task
