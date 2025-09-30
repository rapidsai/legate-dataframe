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

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <legate.h>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/stream_compaction.hpp>
#include <stdexcept>

namespace legate::dataframe {
namespace task {

/*static*/ void ApplyBooleanMaskTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto tbl          = argument::get_next_input<PhysicalTable>(ctx);
  const auto boolean_mask = argument::get_next_input<PhysicalColumn>(ctx);
  auto output             = argument::get_next_output<PhysicalTable>(ctx);

  auto result = ARROW_RESULT(arrow::compute::CallFunction(
                               "filter", {tbl.arrow_table_view(), boolean_mask.arrow_array_view()}))
                  .table();

  output.move_into(std::move(result));
}

/*static*/ void DistinctTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  auto table            = argument::get_next_input<PhysicalTable>(ctx);
  auto output           = argument::get_next_output<PhysicalTable>(ctx);
  auto key_col_idx      = argument::get_next_scalar_vector<int32_t>(ctx);
  auto high_cardinality = argument::get_next_scalar<bool>(ctx);

  auto working_table = table.arrow_table_view();

  // Arrow has no explicit distinct, so we implement it via groupby/hash aggs.
  // Columns that are not keys are aggregated via "hash_one" (pick random entry)
  std::map<std::size_t, std::size_t> key_col_idx_map;
  for (size_t i = 0; i < key_col_idx.size(); ++i) {
    key_col_idx_map[key_col_idx[i]] = i;
  }
  std::vector<arrow::compute::Aggregate> aggregates;
  // The result has agg columns last, so need to reorder before returning.
  std::vector<int> res_col_order(working_table->num_columns());
  for (size_t i = 0; i < working_table->num_columns(); ++i) {
    if (key_col_idx_map.count(i) == 0) {
      res_col_order[i] = key_col_idx.size() + aggregates.size();
      aggregates.push_back({"hash_one", std::to_string(i), std::to_string(i)});
    } else {
      res_col_order[i] = key_col_idx_map.at(i);
    }
  }
  std::vector<arrow::FieldRef> key_names;
  for (auto idx : key_col_idx) {
    key_names.push_back(std::to_string(idx));
  }

  if (!high_cardinality && table.is_partitioned()) {
    arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
      {{"table_source", arrow::acero::TableSourceNodeOptions(working_table)},
       {"aggregate", arrow::acero::AggregateNodeOptions(aggregates, key_names)}});
    working_table =
      ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan), false /*use_threads*/));
  }

  // Repartition `table` based on the keys such that each node can do a local distinct.
  working_table = repartition_by_hash(
    ctx, working_table, std::vector<std::size_t>(key_col_idx.begin(), key_col_idx.end()));

  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(working_table)},
     {"aggregate", arrow::acero::AggregateNodeOptions(aggregates, key_names)}});
  working_table =
    ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan), false /*use_threads*/));
  working_table = ARROW_RESULT(working_table->SelectColumns(res_col_order));

  output.move_into(std::move(working_table));
}

}  // namespace task

LogicalTable apply_boolean_mask(const LogicalTable& tbl, const LogicalColumn& boolean_mask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto ret     = LogicalTable::empty_like(tbl);

  if (boolean_mask.arrow_type() != arrow::boolean()) {
    throw std::invalid_argument("boolean mask column must have a bool dtype.");
  }

  legate::AutoTask task =
    runtime->create_task(get_library(), task::ApplyBooleanMaskTask::TASK_CONFIG.task_id());

  auto tbl_vars = argument::add_next_input(task, tbl);
  auto mask_var = argument::add_next_input(task, boolean_mask);
  argument::add_next_output(task, ret);
  task.add_constraint(legate::align(tbl_vars.at(0), mask_var));

  runtime->submit(std::move(task));
  return ret;
}

LogicalTable distinct(const LogicalTable& tbl,
                      const std::vector<std::string>& keys,
                      bool high_cardinality)
{
  auto runtime = legate::Runtime::get_runtime();
  auto ret     = LogicalTable::empty_like(tbl);

  std::vector<int32_t> key_col_idx;
  for (const std::string& key : keys) {
    key_col_idx.push_back(tbl.get_column_names().at(key));
  }

  legate::AutoTask task =
    runtime->create_task(get_library(), task::DistinctTask::TASK_CONFIG.task_id());

  argument::add_next_input(task, tbl);
  argument::add_next_output(task, ret);
  argument::add_next_scalar_vector(task, key_col_idx);
  argument::add_next_scalar(task, high_cardinality);

  if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    task.add_communicator("cpu");
  } else {
    task.add_communicator("nccl");
  }
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ApplyBooleanMaskTask::register_variants();
  legate::dataframe::task::DistinctTask::register_variants();
}

}  // namespace
