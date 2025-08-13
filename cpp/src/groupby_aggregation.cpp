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

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <legate.h>

#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/groupby_aggregation.hpp>

namespace legate::dataframe {
namespace task {

std::string arrow_aggregation_name(std::string name)
{
  if (name.substr(0, 5) != "hash_") { return "hash_" + name; }
  return name;
}

/*static*/ void GroupByAggregationTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  auto table        = argument::get_next_input<PhysicalTable>(ctx);
  auto output       = argument::get_next_output<PhysicalTable>(ctx);
  auto _key_col_idx = argument::get_next_scalar_vector<size_t>(ctx);
  std::vector<cudf::size_type> key_col_idx(_key_col_idx.begin(), _key_col_idx.end());

  // Get the `column_aggs` task argument
  std::vector<arrow::compute::Aggregate> aggregates;
  auto column_aggs_size = argument::get_next_scalar<size_t>(ctx);
  for (size_t i = 0; i < column_aggs_size; ++i) {
    auto in_col_idx  = argument::get_next_scalar<size_t>(ctx);
    auto kind        = argument::get_next_scalar<std::string>(ctx);
    auto out_col_idx = argument::get_next_scalar<size_t>(ctx);
    aggregates.push_back(
      {arrow_aggregation_name(kind), std::to_string(in_col_idx), std::to_string(i)});
  }

  std::vector<std::string> dummy_column_names;
  for (int i = 0; i < table.num_columns(); i++) {
    dummy_column_names.push_back(std::to_string(i));
  }
  auto table_view = table.arrow_table_view(dummy_column_names);

  // Repartition `table` based on the keys such that each node can do a local groupby.
  auto repartitioned = repartition_by_hash(ctx, table_view, key_col_idx);

  // Do the groupby
  std::vector<arrow::FieldRef> key_names;
  for (auto idx : key_col_idx) {
    key_names.push_back(std::to_string(idx));
  }
  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(repartitioned)},
     {"aggregate", arrow::acero::AggregateNodeOptions(aggregates, key_names)}});
  auto result = ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan)));

  output.move_into(std::move(result));
}

}  // namespace task

namespace {

LogicalColumn make_output_column(const LogicalColumn& values, std::string aggregation_kind)
{
  // Run a dummy arrow aggregation to get the output type
  auto table = arrow::Table::Make(arrow::schema({arrow::field("keys", arrow::int32()),
                                                 arrow::field("values", values.arrow_type())}),
                                  {ARROW_RESULT(arrow::MakeEmptyArray(arrow::int32())),
                                   ARROW_RESULT(arrow::MakeEmptyArray(values.arrow_type()))});

  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(table)},
     {"aggregate",
      arrow::acero::AggregateNodeOptions(
        {arrow::compute::Aggregate(
          task::arrow_aggregation_name(aggregation_kind), {"values"}, "result")},
        {"keys", "values"})}});
  auto result = ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan)));
  // Note: Left nullable here as true - not sure there is a way to know in advance if it should
  // be nullable or not. The safe option is to set it true always.
  return LogicalColumn::empty_like(to_cudf_type(result->column(2)->type()), /* nullable = */ true);
}
}  // namespace

LogicalTable groupby_aggregation(
  const LogicalTable& table,
  const std::vector<std::string>& keys,
  const std::vector<std::tuple<std::string, std::string, std::string>>&  // input column name,
                                                                         // aggregation kind, output
                                                                         // column name
    column_aggregations)
{
  // Let's create the output table
  std::vector<LogicalColumn> output_columns;
  std::vector<std::string> output_column_names;
  // The output table has the key columns first.
  for (std::string key : keys) {
    output_columns.push_back(LogicalColumn::empty_like(table.get_column(key)));
    output_column_names.push_back(std::move(key));
  }
  // And then it has one column per column aggregation
  for (const auto& [in_col_name, kind, out_col_name] : column_aggregations) {
    output_columns.push_back(make_output_column(table.get_column(in_col_name), kind));

    if (std::find(output_column_names.begin(), output_column_names.end(), out_col_name) !=
        output_column_names.end()) {
      throw std::invalid_argument("name conflict in the output columns: " + out_col_name);
    } else {
      output_column_names.push_back(out_col_name);
    }
  }
  LogicalTable output(std::move(output_columns), std::move(output_column_names));

  // Since `PhysicalTable` doesn't have column names, we convert names to indices
  std::vector<std::tuple<size_t, std::string, size_t>> column_aggs;
  for (const auto& [in_col_name, kind, out_col_name] : column_aggregations) {
    size_t in_col_idx = table.get_column_names().at(in_col_name);
    // We index the output columns after the key columns
    size_t out_col_idx = keys.size() + column_aggs.size();
    column_aggs.push_back(std::make_tuple(in_col_idx, kind, out_col_idx));
  }
  std::vector<size_t> key_col_idx;
  for (const std::string& key : keys) {
    key_col_idx.push_back(table.get_column_names().at(key));
  }

  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::GroupByAggregationTask::TASK_CONFIG.task_id());
  argument::add_next_input(task, table);
  argument::add_next_output(task, output);
  argument::add_next_scalar_vector(task, key_col_idx);
  // Add the `column_aggs` task argument
  argument::add_next_scalar(task, column_aggs.size());
  for (const auto& [in_col_idx, kind, out_col_idx] : column_aggs) {
    argument::add_next_scalar(task, in_col_idx);
    argument::add_next_scalar(task, kind);
    argument::add_next_scalar(task, out_col_idx);
  }

  if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    task.add_communicator("cpu");
  } else {
    task.add_communicator("nccl");
  }
  runtime->submit(std::move(task));
  return output;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::GroupByAggregationTask::register_variants();
}

}  // namespace
