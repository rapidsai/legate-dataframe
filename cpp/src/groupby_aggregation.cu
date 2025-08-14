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

#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/groupby.hpp>
#include <cudf/unary.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/groupby_aggregation.hpp>

namespace legate::dataframe::task {

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
    case cudf::aggregation::Kind::COUNT_VALID: {
      return cudf::make_count_aggregation<cudf::groupby_aggregation>();
    }
    case cudf::aggregation::Kind::MEAN: {
      return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    }
    // 0 degrees of freedom instead of default 1 to match Arrow's behavior
    case cudf::aggregation::Kind::VARIANCE: {
      return cudf::make_variance_aggregation<cudf::groupby_aggregation>(0);
    }
    case cudf::aggregation::Kind::STD: {
      return cudf::make_std_aggregation<cudf::groupby_aggregation>(0);
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

cudf::aggregation::Kind arrow_to_cudf_aggregation(const std::string& agg_name)
{
  std::map<std::string, cudf::aggregation::Kind> agg_map = {
    // Direct mappings
    {"sum", cudf::aggregation::Kind::SUM},
    {"product", cudf::aggregation::Kind::PRODUCT},
    {"min", cudf::aggregation::Kind::MIN},
    {"max", cudf::aggregation::Kind::MAX},
    {"count", cudf::aggregation::Kind::COUNT_VALID},
    {"mean", cudf::aggregation::Kind::MEAN},
    {"variance", cudf::aggregation::Kind::VARIANCE},
    {"stddev", cudf::aggregation::Kind::STD},
    {"approximate_median", cudf::aggregation::Kind::MEDIAN},
    {"count_distinct", cudf::aggregation::Kind::NUNIQUE}};

  //  {"count_all", cudf::aggregation::Kind::COUNT_ALL},
  // "count_all" could be supported but needs some work as it has 0 inputs

  // Don't do these as we don't support nested types at the moment
  // {"list", cudf::aggregation::Kind::COLLECT_LIST},
  // {"tdigest", cudf::aggregation::Kind::TDIGEST}

  // Arrow aggregations with no direct cuDF equivalent:
  // any - cudf has as a reduction aggregation but not groupby aggregation
  // all - cudf has as a reduction aggregation but not groupby aggregation
  // distinct
  // first - could map to NTH_ELEMENT with n=0
  // first_last - no equivalent
  // kurtosis - no equivalent
  // last - could map to NTH_ELEMENT with n=-1
  // min_max - no single equivalent (would need separate MIN/MAX)
  // one - no equivalent
  // pivot_wider - no equivalent
  // skew - no equivalent

  // cuDF aggregations with no direct Arrow equivalent:
  // SUM_OF_SQUARES - no equivalent
  // M2 - no equivalent
  // QUANTILE - no equivalent
  // ARGMAX - no equivalent
  // ARGMIN - no equivalent
  // NTH_ELEMENT - no equivalent
  // ROW_NUMBER - no equivalent
  // EWMA - no equivalent
  // RANK - no equivalent
  // COLLECT_SET - no equivalent
  // LEAD - no equivalent
  // LAG - no equivalent
  // PTX - no equivalent
  // CUDA - no equivalent
  // HOST_UDF - no equivalent
  // MERGE_LISTS - no equivalent
  // MERGE_SETS - no equivalent
  // MERGE_M2 - no equivalent
  // COVARIANCE - no equivalent
  // CORRELATION - no equivalent
  // MERGE_TDIGEST - no equivalent
  // HISTOGRAM - no equivalent
  // MERGE_HISTOGRAM - no equivalent
  // BITWISE_AGG - no equivalent
  if (agg_map.count(agg_name) == 0) {
    throw std::invalid_argument("Unsupported aggregation: " + agg_name);
  }
  return agg_map.at(agg_name);
}

/*static*/ void GroupByAggregationTask::gpu_variant(legate::TaskContext context)
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
    auto kind        = argument::get_next_scalar<std::string>(ctx);
    auto out_col_idx = argument::get_next_scalar<size_t>(ctx);
    column_aggs.push_back({in_col_idx, arrow_to_cudf_aggregation(kind), out_col_idx});
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
      out_col_to_request_and_agg_idx[out_col_idx] = {request_idx, request.aggregations.size() - 1};
    }
  }

  // Do a local groupby
  cudf::groupby::groupby gb_obj(repartitioned->select(key_col_idx), cudf::null_policy::INCLUDE);
  auto [unique_keys, agg_result] = gb_obj.aggregate(requests, ctx.stream(), ctx.mr());

  // Gather the output columns. The key columns goes first.
  auto output_columns = unique_keys->release();

  // Then we add the columns in `agg_result` using the order recorded
  // in `out_col_to_request_and_agg_idx`.
  output_columns.resize(output_columns.size() + out_col_to_request_and_agg_idx.size());
  auto out_types = output.cudf_types();
  for (auto [out_col_idx, request_and_agg_idx] : out_col_to_request_and_agg_idx) {
    auto [request_idx, agg_idx] = request_and_agg_idx;

    output_columns.at(out_col_idx) = std::move(agg_result.at(request_idx).results.at(agg_idx));

    // Cast the cudf output to be consistent with the output, which has output types according to
    // arrow convention
    if (output_columns.at(out_col_idx)->type() != out_types.at(out_col_idx)) {
      output_columns.at(out_col_idx) = cudf::cast(
        output_columns.at(out_col_idx)->view(), out_types.at(out_col_idx), ctx.stream(), ctx.mr());
    }
  }

  output.move_into(std::move(output_columns));
}
}  // namespace legate::dataframe::task
