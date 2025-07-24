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

#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/groupby.hpp>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/groupby_aggregation.hpp>

namespace legate::dataframe {
namespace task {

std::string cudf_to_arrow_aggregate_type(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: return "hash_sum";
    case cudf::aggregation::Kind::NUNIQUE: return "hash_count_distinct";
    case cudf::aggregation::Kind::MAX: return "hash_max";
    case cudf::aggregation::Kind::MIN: return "hash_min";
    case cudf::aggregation::Kind::MEAN: return "hash_mean";
    case cudf::aggregation::Kind::MEDIAN: return "hash_approximate_median";
    default: throw std::invalid_argument("Unsupported aggregation kind");
  }
}

class GroupByAggregationTask : public Task<GroupByAggregationTask, OpCode::GroupByAggregation> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::GroupByAggregation}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(true)
                                                .with_elide_device_ctx_sync(true);
  static void cpu_variant(legate::TaskContext context)
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
      auto kind        = argument::get_next_scalar<cudf::aggregation::Kind>(ctx);
      auto out_col_idx = argument::get_next_scalar<size_t>(ctx);
      aggregates.push_back(
        {cudf_to_arrow_aggregate_type(kind), std::to_string(in_col_idx), std::to_string(i)});
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

    // Make sure the columns match the output type
    // TODO: this may not be necessary if we create the column types to match arrow
    auto expected_arrow_types = output.arrow_types();
    for (size_t i = 0; i < result->num_columns(); ++i) {
      auto col = result->column(i);
      if (col->type() != expected_arrow_types.at(i) && col->num_chunks() > 0) {
        auto cast = ARROW_RESULT(arrow::compute::Cast(*arrow::Concatenate(col->chunks()),
                                                      expected_arrow_types.at(i)))
                      .make_array();
        auto field = arrow::field(std::to_string(i), expected_arrow_types.at(i));
        result =
          ARROW_RESULT(result->SetColumn(i, field, std::make_shared<arrow::ChunkedArray>(cast)));
      }
    }

    output.move_into(std::move(result));
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    auto table        = argument::get_next_input<PhysicalTable>(ctx);
    auto output       = argument::get_next_output<PhysicalTable>(ctx);
    auto _key_col_idx = argument::get_next_scalar_vector<size_t>(ctx);
    std::vector<cudf::size_type> key_col_idx(_key_col_idx.begin(), _key_col_idx.end());

    // Get the `column_aggs` task argument
    std::vector<std::tuple<size_t, cudf::aggregation::Kind, size_t>> column_aggs;
    auto column_aggs_size = argument::get_next_scalar<size_t>(ctx);
    for (size_t i = 0; i < column_aggs_size; ++i) {
      auto in_col_idx  = argument::get_next_scalar<size_t>(ctx);
      auto kind        = argument::get_next_scalar<cudf::aggregation::Kind>(ctx);
      auto out_col_idx = argument::get_next_scalar<size_t>(ctx);
      column_aggs.push_back({in_col_idx, kind, out_col_idx});
    }

    // Repartition `table` based on the keys such that each node can do a local groupby.
    auto repartitioned = repartition_by_hash(ctx, table.table_view(), key_col_idx);

    // In order to create the aggregation requests, we walk through `column_aggs` and for
    // each unique input-column-index, we create an aggregation request and append the
    // aggregation-kinds found in `column_aggs`.
    std::vector<cudf::groupby::aggregation_request> requests;
    std::map<size_t, std::pair<size_t, size_t>> out_col_to_request_and_agg_idx;
    {
      std::map<size_t, size_t> in_col_to_request_idx;
      for (const auto& [in_col_idx, kind, out_col_idx] : column_aggs) {
        // If this is the first time we see `in_col_idx`, we create a new `aggregation_request`
        // with `values` set to the column of `in_col_idx` and an empty aggregation vector.
        if (in_col_to_request_idx.find(in_col_idx) == in_col_to_request_idx.end()) {
          in_col_to_request_idx[in_col_idx] = requests.size();
          requests.push_back(cudf::groupby::aggregation_request{
            .values = repartitioned->get_column(in_col_idx), .aggregations = {}});
        }

        // Find the `aggregation_request` that belongs to `in_col_idx`
        size_t request_idx = in_col_to_request_idx.at(in_col_idx);
        auto& request      = requests.at(request_idx);
        // Add the aggregation kind to the request
        request.aggregations.push_back(make_groupby_aggregation(kind));

        // Record in which index in `requests` and `request.aggregations`, the
        // aggregation was added.
        out_col_to_request_and_agg_idx[out_col_idx] = {request_idx,
                                                       request.aggregations.size() - 1};
      }
    }

    // Do a local groupby
    cudf::groupby::groupby gb_obj(repartitioned->select(key_col_idx));
    auto [unique_keys, agg_result] = gb_obj.aggregate(requests, ctx.stream(), ctx.mr());

    // Gather the output columns. The key columns goes first.
    auto output_columns = unique_keys->release();

    // Then we add the columns in `agg_result` using the order recorded
    // in `out_col_to_request_and_agg_idx`.
    output_columns.resize(output_columns.size() + out_col_to_request_and_agg_idx.size());
    for (auto [out_col_idx, request_and_agg_idx] : out_col_to_request_and_agg_idx) {
      auto [request_idx, agg_idx]    = request_and_agg_idx;
      output_columns.at(out_col_idx) = std::move(agg_result.at(request_idx).results.at(agg_idx));
    }
    output.move_into(std::move(output_columns));
  }
};

}  // namespace task

std::unique_ptr<cudf::groupby_aggregation> make_groupby_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: {
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::PRODUCT: {
      return cudf::make_product_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::MIN: {
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::MAX: {
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::SUM_OF_SQUARES: {
      return cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::MEAN: {
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::STD: {
      return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::VARIANCE: {
      return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::MEDIAN: {
      return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::NUNIQUE: {
      return cudf::make_nunique_aggregation<cudf::groupby_aggregation>();
    }
    default: {
      throw std::invalid_argument("Unsupported groupby aggregation");
    }
  }
}

std::vector<std::unique_ptr<cudf::groupby_aggregation>> make_groupby_aggregations(
  const std::vector<cudf::aggregation::Kind>& kinds)
{
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> ret;
  for (auto kind : kinds) {
    ret.push_back(make_groupby_aggregation(kind));
  }
  return ret;
}

namespace {
LogicalColumn make_output_column(const LogicalColumn& values, cudf::aggregation::Kind kind)
{
  auto dtype = cudf::detail::target_type(values.cudf_type(), kind);

  switch (kind) {
    case cudf::aggregation::Kind::SUM:
    case cudf::aggregation::Kind::PRODUCT:
    case cudf::aggregation::Kind::MIN:
    case cudf::aggregation::Kind::MAX:
    case cudf::aggregation::Kind::SUM_OF_SQUARES:
    case cudf::aggregation::Kind::MEAN:
    case cudf::aggregation::Kind::NUNIQUE: {
      return LogicalColumn::empty_like(dtype, values.nullable());
    }
    case cudf::aggregation::Kind::STD:
    case cudf::aggregation::Kind::VARIANCE:
    case cudf::aggregation::Kind::MEDIAN: {
      // These might be nullable even when the input isn't
      return LogicalColumn::empty_like(dtype, /* nullable = */ true);
    }
    default: {
      throw std::invalid_argument("Unsupported groupby aggregation");
    }
  }
}
}  // namespace

LogicalTable groupby_aggregation(
  const LogicalTable& table,
  const std::vector<std::string>& keys,
  const std::vector<std::tuple<std::string, cudf::aggregation::Kind, std::string>>&
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
  std::vector<std::tuple<size_t, cudf::aggregation::Kind, size_t>> column_aggs;
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
    argument::add_next_scalar(task,
                              static_cast<std::underlying_type_t<cudf::aggregation::Kind>>(kind));
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

const auto reg_id_ = []() -> char {
  legate::dataframe::task::GroupByAggregationTask::register_variants();
  return 0;
}();

}  // namespace
