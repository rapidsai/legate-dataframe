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

#include <numeric>
#include <stdexcept>
#include <vector>

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/merge.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <legate.h>
#include <legate/cuda/cuda.h>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/join.hpp>

#define DEBUG_SPLITS 0
#if DEBUG_SPLITS
#include <iostream>
#include <sstream>
#endif

namespace legate::dataframe {
namespace task {
namespace {

/**
 * @brief Return points at which to split a dataset.
 *
 * @param nvalues The total number of values to split.
 * @param nsplits the number of splits (and split values as last is included)
 * @param include_start Whether to include the starting 0.
 * @returns cudf column selecting containing nsplits indices.
 */
std::vector<cudf::size_type> get_split_ind(TaskContext& ctx,
                                           cudf::size_type nvalues,
                                           int nsplits,
                                           bool include_start)
{
  cudf::size_type nvalues_per_split, nvalues_left;
  if (nvalues < nsplits) {
    // Only return non-empty splits (we could return one point multiple times)
    nsplits           = nvalues;
    nvalues_per_split = (nsplits > 0);
    nvalues_left      = 0;
  } else {
    nvalues_per_split = nvalues / nsplits;
    nvalues_left      = nvalues - nvalues_per_split * nsplits;
  }

  std::vector<cudf::size_type> split_values;
  cudf::size_type split_offset = 0;

  if (include_start && nvalues > 0) { split_values.push_back(0); }

  for (cudf::size_type i = 0; i < nsplits - 1; i++) {
    split_offset += nvalues_per_split;
    if (i < nvalues_left) { split_offset += 1; }

    split_values.push_back(split_offset);
  }
  assert(split_offset + nvalues_per_split == nvalues);

#if DEBUG_SPLITS
  std::ostringstream splits_points_oss;
  splits_points_oss << "Split points @" << nvalues << ": ";
  for (auto point : split_values) {
    splits_points_oss << point << ", ";
  }
  std::cout << splits_points_oss.str() << std::endl;
#endif

  return split_values;
}

}  // namespace

namespace cpu {

template <typename IntT>
std::shared_ptr<arrow::Array> vector_to_array(std::vector<IntT>&& vec)
{
  arrow::Int64Builder builder;
  auto status = builder.AppendValues(vec.begin(), vec.end());
  return ARROW_RESULT(builder.Finish());
}

template <typename T>
std::shared_ptr<arrow::Array> create_array(int64_t num_elements, T fill_value)
{
  using BuilderType =
    typename arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::BuilderType;
  BuilderType builder;
  builder.Reserve(num_elements);
  for (int64_t i = 0; i < num_elements; i++) {
    builder.UnsafeAppend(fill_value);
  }
  return ARROW_RESULT(builder.Finish());
}

std::shared_ptr<arrow::Table> extract_local_splits(TaskContext& ctx,
                                                   std::shared_ptr<arrow::Table> sorted_table,
                                                   const std::vector<cudf::size_type>& keys_idx)
{
  auto split_indices =
    vector_to_array(get_split_ind(ctx, sorted_table->num_rows(), ctx.nranks, true));

  auto take_splits = ARROW_RESULT(arrow::compute::Take(sorted_table, split_indices)).table();

  auto split_ranks = vector_to_array(std::vector<int64_t>(split_indices->length(), ctx.rank));

  auto local_splits_and_metadata =
    ARROW_RESULT(take_splits->AddColumn(take_splits->num_columns(),
                                        arrow::field("split_rank", split_ranks->type()),
                                        std::make_shared<arrow::ChunkedArray>(split_ranks)));

  local_splits_and_metadata = ARROW_RESULT(
    local_splits_and_metadata->AddColumn(local_splits_and_metadata->num_columns(),
                                         arrow::field("split_index", split_indices->type()),
                                         std::make_shared<arrow::ChunkedArray>(split_indices)));

  return local_splits_and_metadata;
}

std::shared_ptr<arrow::Table> merge_distributed_split_candidates(
  TaskContext& ctx,
  std::shared_ptr<arrow::Table> local_splits_and_metadata,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<arrow::compute::SortKey>& column_order,
  arrow::compute::NullPlacement null_precedence)
{
  std::vector<std::shared_ptr<arrow::Table>> exchange_tables;
  for (int i = 0; i < ctx.nranks; i++) {
    exchange_tables.push_back(local_splits_and_metadata);
  }
  auto shuffled = shuffle(ctx, exchange_tables);

  if (local_splits_and_metadata->num_rows() == 0) {
    // All nodes need to take part in the shuffle (no data here), but the below
    // cannot search a length 0 table, so return immediately.
    return nullptr;
  }

  auto all_split_candidates = ARROW_RESULT(arrow::ConcatenateTables(shuffled));

  // TODO: Add the rank column as a sort key?
  // Potentially we don't need this as a stable sort would maintain rank order for equal rows
  arrow::compute::SortOptions sort_options(column_order, null_precedence);
  auto sorted_indices =
    ARROW_RESULT(arrow::compute::SortIndices(all_split_candidates, sort_options));
  auto sorted_table =
    ARROW_RESULT(
      arrow::compute::Take(all_split_candidates, *sorted_indices, arrow::compute::TakeOptions{}))
      .table();
  return sorted_table;
}

std::shared_ptr<arrow::Table> extract_global_splits(
  TaskContext& ctx, std::shared_ptr<arrow::Table> global_split_candidates)
{
  auto split_indices = vector_to_array(
    get_split_ind(ctx, global_split_candidates->num_rows(), ctx.nranks, /* include_start */ false));
  auto split_values =
    ARROW_RESULT(arrow::compute::Take(global_split_candidates, split_indices)).table();
  return std::move(split_values);
}

// This function is poorly optimised but arrow doesn't give us other options
// This comparison must behave exactly the same as arrow's SortIndices
bool compare(TaskContext& ctx,
             const arrow::Table& table_a,
             const arrow::Table& table_b,
             std::size_t row_index_a,
             std::size_t row_index_b,
             const std::vector<cudf::size_type>& keys_idx,
             const std::vector<arrow::compute::SortKey>& column_order,
             arrow::compute::NullPlacement null_precedence)
{
  for (std::size_t i = 0; i < keys_idx.size(); i++) {
    auto col_a = table_a.column(keys_idx[i]);
    auto col_b = table_b.column(keys_idx[i]);

    auto value_a = ARROW_RESULT(col_a->GetScalar(row_index_a));
    auto value_b = ARROW_RESULT(col_b->GetScalar(row_index_b));

    auto a_null = !(*value_a).is_valid;
    auto b_null = !(*value_b).is_valid;

    if (a_null && b_null) continue;

    if (a_null) return null_precedence == arrow::compute::NullPlacement::AtStart;
    if (b_null) return null_precedence == arrow::compute::NullPlacement::AtEnd;

    if (value_a->Equals(*value_b)) continue;  // Move to next column if equal

    auto compare_function =
      column_order.at(i).order == arrow::compute::SortOrder::Ascending ? "less" : "greater";
    auto result = ARROW_RESULT(arrow::compute::CallFunction(compare_function, {value_a, value_b}))
                    .scalar_as<arrow::BooleanScalar>()
                    .value;

    return result;  // Return true if value_a < value_b
  }
  return false;  // Equal values
}

std::size_t lower_bound_row(TaskContext& ctx,
                            const arrow::Table& sorted_table_with_rank,
                            const arrow::Table& global_split_values,
                            std::size_t split_index,
                            const std::vector<cudf::size_type>& keys_idx,
                            const std::vector<arrow::compute::SortKey>& column_order,
                            arrow::compute::NullPlacement null_precedence)
{
  // Linear search version
  for (std::size_t i = 0; i < sorted_table_with_rank.num_rows(); i++) {
    if (compare(ctx,
                global_split_values,
                sorted_table_with_rank,
                split_index,
                i,
                keys_idx,
                column_order,
                null_precedence)) {
      return i;
    }
  }

  return sorted_table_with_rank.num_rows();  // Not found, return end
}

std::vector<std::size_t> find_destination_ranks(
  TaskContext& ctx,
  std::shared_ptr<arrow::Table> sorted_table,
  std::shared_ptr<arrow::Table> global_split_values,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<arrow::compute::SortKey>& column_order,
  arrow::compute::NullPlacement null_precedence

)
{
  // Create a new table with the rank column appended
  auto rank_array = create_array<int32_t>(sorted_table->num_rows(), ctx.rank);
  auto sorted_table_with_rank =
    ARROW_RESULT(sorted_table->AddColumn(sorted_table->num_columns(),
                                         arrow::field("rank", rank_array->type()),
                                         std::make_shared<arrow::ChunkedArray>(rank_array)));

  std::vector<std::size_t> splits_indices_host;

  auto keys_idx_with_rank = keys_idx;
  keys_idx_with_rank.push_back(sorted_table->num_columns());
  auto column_order_with_rank = column_order;
  column_order_with_rank.push_back(
    arrow::compute::SortKey{"split_rank", arrow::compute::SortOrder::Ascending});

  // For each global split value, find where it should be inserted in the local sorted table
  for (std::size_t i = 0; i < static_cast<std::size_t>(global_split_values->num_rows()); i++) {
    std::size_t position = lower_bound_row(ctx,
                                           *sorted_table_with_rank,
                                           *global_split_values,
                                           i,
                                           keys_idx_with_rank,
                                           column_order_with_rank,
                                           null_precedence);
    splits_indices_host.push_back(position);
  }

  // In the obscure case where there is less data than ranks, pad split points.
  for (std::size_t i = splits_indices_host.size(); i < static_cast<std::size_t>(ctx.nranks - 1);
       i++) {
    splits_indices_host.push_back(static_cast<std::size_t>(sorted_table->num_rows()));
  }

  if (!std::is_sorted(splits_indices_host.begin(), splits_indices_host.end())) {
    throw std::runtime_error(
      "Splits indices should be sorted. This is a bug, and indicates a difference between arrow's "
      "sort comparator and the custom comparator used in this file.");
  }

  return splits_indices_host;
}

std::vector<std::size_t> find_splits_for_distribution(
  TaskContext& ctx,
  std::shared_ptr<arrow::Table> sorted_table,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<arrow::compute::SortKey>& column_order,
  arrow::compute::NullPlacement null_precedence)
{
  auto local_splits_and_metadata = extract_local_splits(ctx, sorted_table, keys_idx);

  auto global_split_candidates = merge_distributed_split_candidates(
    ctx, local_splits_and_metadata, keys_idx, column_order, null_precedence);

  if (global_split_candidates == nullptr) {
    // Nothing on this worker, we are done
    return {};
  }

  auto global_split_values = extract_global_splits(ctx, global_split_candidates);

  return find_destination_ranks(
    ctx, sorted_table, global_split_values, keys_idx, column_order, null_precedence);
}

}  // namespace cpu

namespace gpu {
std::unique_ptr<cudf::column> vector_to_column(const std::vector<cudf::size_type>& vec,
                                               TaskContext& ctx)
{
  auto ncopy = vec.size();
  rmm::device_uvector<cudf::size_type> split_ind(ncopy, ctx.stream(), ctx.mr());
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_ind.data(),
                                    vec.data(),
                                    ncopy * sizeof(cudf::size_type),
                                    cudaMemcpyHostToDevice,
                                    ctx.stream()));
  LEGATE_CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));

  return std::make_unique<cudf::column>(std::move(split_ind), std::move(rmm::device_buffer()), 0);
}

// Create a cudf column with the specified number of rows
template <typename T>
std::unique_ptr<cudf::column> create_column(cudf::size_type num_rows,
                                            T fill_value,
                                            TaskContext& ctx)
{
  if (num_rows == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<T>()}); }
  return cudf::sequence(num_rows,
                        *cudf::make_fixed_width_scalar(fill_value),
                        *cudf::make_fixed_width_scalar(int32_t{0}));
}

template <typename T>
std::vector<T> column_to_vector(TaskContext& ctx, const cudf::column_view& col)
{
  std::vector<T> ret(col.size());
  if (col.size() > 0) {
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(
      ret.data(), col.data<T>(), col.size() * sizeof(T), cudaMemcpyDeviceToHost, ctx.stream()));
  }
  return ret;
}

// Extract split points from a sorted table. Add two metadata columns:
//  - the rank of the split point (which worker it came from)
//  - the local index of the split point
std::unique_ptr<cudf::table> extract_local_splits(TaskContext& ctx,
                                                  const cudf::table_view& sorted_table,
                                                  const std::vector<cudf::size_type>& keys_idx)
{
  auto split_values     = get_split_ind(ctx, sorted_table.num_rows(), ctx.nranks, true);
  auto my_split_ind_col = vector_to_column(split_values, ctx);
  auto nsplits          = my_split_ind_col->size();

  auto my_split_rank_col = create_column<int32_t>(nsplits, ctx.rank, ctx);

  auto my_split_cols_tbl = cudf::gather(sorted_table.select(keys_idx),
                                        my_split_ind_col->view(),
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        ctx.stream(),
                                        ctx.mr());
  auto table_columns     = my_split_cols_tbl->release();
  table_columns.push_back(std::move(my_split_rank_col));
  table_columns.push_back(std::move(my_split_ind_col));
  return std::make_unique<cudf::table>(std::move(table_columns));
}

std::unique_ptr<cudf::table> merge_distributed_split_candidates(
  TaskContext& ctx,
  const cudf::table_view& local_splits_and_metadata,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence)
{
  std::vector<cudf::table_view> exchange_tables;
  for (int i = 0; i < ctx.nranks; i++) {
    exchange_tables.push_back(local_splits_and_metadata);
  }
  auto [split_candidates_shared, owners_split] = shuffle(ctx, exchange_tables, nullptr);

  if (local_splits_and_metadata.num_rows() == 0) {
    // All nodes need to take part in the shuffle (no data here), but the below
    // cannot search a length 0 table, so return immediately.
    return nullptr;
  }

  std::vector<cudf::order> column_orderx(column_order);
  std::vector<cudf::null_order> null_precedencex(null_precedence);
  column_orderx.insert(column_orderx.end(), {cudf::order::ASCENDING, cudf::order::ASCENDING});
  null_precedencex.insert(null_precedencex.end(),
                          {cudf::null_order::AFTER, cudf::null_order::AFTER});

  // Merge is stable as it includes the rank and index in the keys:
  // keys(x) to pick columns from splits (which include rank and index):
  std::vector<cudf::size_type> all_keysx(keys_idx.size() + 2);
  std::iota(all_keysx.begin(), all_keysx.end(), 0);

  auto split_candidates = cudf::merge(
    split_candidates_shared, all_keysx, column_orderx, null_precedencex, ctx.stream(), ctx.mr());
  owners_split.reset();  // No longer need this
  return std::move(split_candidates);
}

std::unique_ptr<cudf::table> extract_global_splits(TaskContext& ctx,
                                                   const cudf::table_view& global_split_candidates)
{
  auto split_indices =
    get_split_ind(ctx, global_split_candidates.num_rows(), ctx.nranks, /* include_start */ false);
  auto split_value_inds = vector_to_column(split_indices, ctx);
  auto split_values     = cudf::gather(global_split_candidates,
                                   split_value_inds->view(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   ctx.stream(),
                                   ctx.mr());
  return std::move(split_values);
}

std::vector<cudf::size_type> find_destination_ranks(
  TaskContext& ctx,
  const cudf::table_view& sorted_table,
  const cudf::table_view& global_split_values,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence

)
{
  std::vector<cudf::size_type> value_keysx(keys_idx.size() + 1);
  std::iota(value_keysx.begin(), value_keysx.end(), 0);
  auto keys_idxx = keys_idx;
  keys_idxx.push_back(sorted_table.num_columns());

  // Create a column with the same length as sorted table, filled with current rank
  auto rank_column = create_column<int32_t>(sorted_table.num_rows(), ctx.rank, ctx);

  // Create a new table view by appending the rank column to the sorted table
  std::vector<cudf::column_view> table_columns;
  for (int i = 0; i < sorted_table.num_columns(); i++) {
    table_columns.push_back(sorted_table.column(i));
  }
  table_columns.push_back(rank_column->view());
  auto sorted_table_with_rank = cudf::table_view(table_columns);

  auto split_indices = cudf::lower_bound(sorted_table_with_rank.select(keys_idxx),
                                         global_split_values.select(value_keysx),
                                         column_order,
                                         null_precedence,
                                         ctx.stream(),
                                         ctx.mr());

  /*
   * Copy the split candidates to the host and finalize the local splits.
   * (we may have fewer than nranks split-points here and need to pad later.)
   */
  auto splits_indices_host = column_to_vector<cudf::size_type>(ctx, split_indices->view());
  LEGATE_CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));
  // In the obscure case where there is less data than ranks, pad split points.
  for (int i = splits_indices_host.size(); i < ctx.nranks - 1; i++) {
    splits_indices_host.push_back(sorted_table.num_rows());
  }

  return splits_indices_host;
}

/*
 * The practical way to do a distributed sort is to use the initial locally
 * sorted table to estimate good split points to shuffle data to the final node.
 *
 * The rough approach for shuffling the data is the following:
 * 1. Extract `nranks` split candidates from the local table and add their rank
 *    and local index.
 * 2. Exchange all split candidate values and sort them
 * 3. Again extract those candidates that evenly split the whole candidate set.
 *    (we do this on all nodes).
 * 4. Shuffle the data based on the final split candidates.
 *
 * This approach is e.g. the same as in cupynumeric.  We cannot guarantee balanced
 * result chunk sizes, but it should ensure results are within 2x the input chunks.
 * If all chunks are balanced and have the same distribution, the result will be
 * (approximately) balanced again.
 *
 * The trickiest thing to take care of are equal values.  Depending which rank
 * the split point came from (i.e. where it is globally from us), we need to pick
 * the split point inde (if ours) or the first equal value or just after the last
 * depending on whether it came from an earlier or later rank.
 */
std::vector<cudf::size_type> find_splits_for_distribution(
  TaskContext& ctx,
  const cudf::table_view& sorted_table,
  const std::vector<cudf::size_type>& keys_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence)
{
  /*
   * Step 1: Extract local candidates and add rank and index information.
   *
   * We use the start index to find the value representing the range
   * (used as a possible split value), but store the corresponding end of the
   * the last step.
   */
  auto local_splits_and_metadata = extract_local_splits(ctx, sorted_table, keys_idx);

  /*
   * Step 2: Share split candidates among all ranks.
   */
  auto global_split_candidates = merge_distributed_split_candidates(
    ctx, local_splits_and_metadata->view(), keys_idx, column_order, null_precedence);

  if (global_split_candidates == nullptr) {
    // Nothing on this worker, we are done
    return {};
  }

  /*
   * Step 3: Find the best splitting points from all candidates
   */
  auto global_split_values = extract_global_splits(ctx, global_split_candidates->view());

  /*
   * Step 4: Find the actual split points for the local dataset.
   *
   */
  return find_destination_ranks(
    ctx, sorted_table, global_split_values->view(), keys_idx, column_order, null_precedence);
}

}  // namespace gpu

class SortTask : public Task<SortTask, OpCode::Sort> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::Sort}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(true)
                                                .with_elide_device_ctx_sync(true);

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto tbl            = argument::get_next_input<PhysicalTable>(ctx);
    const auto keys_idx       = argument::get_next_scalar_vector<cudf::size_type>(ctx);
    const auto sort_ascending = argument::get_next_scalar_vector<bool>(ctx);
    const auto nulls_at_end   = argument::get_next_scalar<bool>(ctx);
    const auto stable         = argument::get_next_scalar<bool>(ctx);
    auto output               = argument::get_next_output<PhysicalTable>(ctx);

    // Sort the table
    // Use integer indices as arrow column names
    std::vector<std::string> tmp_column_names;
    for (auto idx = 0; idx < tbl.num_columns(); idx++) {
      tmp_column_names.push_back(std::to_string(idx));
    }
    std::vector<arrow::compute::SortKey> sort_keys;
    for (size_t i = 0; i < keys_idx.size(); i++) {
      // Translate cudf parameters to arrow parameters
      auto order = sort_ascending[i] ? arrow::compute::SortOrder::Ascending
                                     : arrow::compute::SortOrder::Descending;
      sort_keys.push_back(arrow::compute::SortKey{std::to_string(keys_idx[i]), order});
    }
    auto arrow_table = tbl.arrow_table_view(tmp_column_names);
    // Arrow does not support null_order per column, so we use the first one
    auto null_order =
      nulls_at_end ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart;
    arrow::compute::SortOptions sort_options(sort_keys, null_order);
    auto sorted_indices = ARROW_RESULT(arrow::compute::SortIndices(arrow_table, sort_options));
    auto sorted_table   = ARROW_RESULT(arrow::compute::Take(
                                       arrow_table, *sorted_indices, arrow::compute::TakeOptions{}))
                          .table();

    if (ctx.nranks == 1) {
      output.move_into(sorted_table);
      return;
    }

    auto split_indices =
      cpu::find_splits_for_distribution(ctx, sorted_table, keys_idx, sort_keys, null_order);

    // If the local table has zero rows we cannot split it for sharing and
    // split_indices will be null.  Exchange the (empty) table instead.
    std::vector<std::shared_ptr<arrow::Table>> partitions;
    if (split_indices.size() > 0) {
      partitions.push_back(sorted_table->Slice(0, split_indices[0]));
      for (int i = 1; i < split_indices.size() + 1; i++) {
        auto start = split_indices[i - 1];
        auto length =
          (i == split_indices.size()) ? sorted_table->num_rows() - start : split_indices[i] - start;
        partitions.push_back(sorted_table->Slice(start, length));
      }
    } else {
      assert(sorted_table->num_rows() == 0);
      for (int i = 0; i < ctx.nranks; i++) {
        partitions.push_back(sorted_table);
      }
    }
    auto parts = shuffle(ctx, partitions);

    // Concateate results and sort
    auto concatenated = ARROW_RESULT(arrow::ConcatenateTables(parts));
    auto sort_indices = ARROW_RESULT(arrow::compute::SortIndices(concatenated, sort_options));
    auto result       = ARROW_RESULT(arrow::compute::Take(concatenated, *sort_indices)).table();
    output.move_into(result);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto tbl            = argument::get_next_input<PhysicalTable>(ctx);
    const auto keys_idx       = argument::get_next_scalar_vector<cudf::size_type>(ctx);
    const auto sort_ascending = argument::get_next_scalar_vector<bool>(ctx);
    const auto nulls_at_end   = argument::get_next_scalar<bool>(ctx);
    const auto stable         = argument::get_next_scalar<bool>(ctx);
    auto output               = argument::get_next_output<PhysicalTable>(ctx);

    // Convert ordering parameters to cudf types
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;
    for (size_t i = 0; i < keys_idx.size(); i++) {
      column_order.push_back(sort_ascending[i] ? cudf::order::ASCENDING : cudf::order::DESCENDING);
      // Flip the null order if the column is descending
      // This makes the result consistent with arrow
      // Otherwise cudf will put nulls at the start of descending columns with
      // cudf::null_order::AFTER
      if (sort_ascending[i] == false) {
        null_precedence.push_back(nulls_at_end ? cudf::null_order::BEFORE
                                               : cudf::null_order::AFTER);
      } else {
        null_precedence.push_back(nulls_at_end ? cudf::null_order::AFTER
                                               : cudf::null_order::BEFORE);
      }
    }

    // Create a new locally sorted table (we always need this)
    auto cudf_tbl  = tbl.table_view();
    auto key       = cudf_tbl.select(keys_idx);
    auto sort_func = stable ? cudf::stable_sort_by_key : cudf::sort_by_key;
    auto sorted_table =
      sort_func(cudf_tbl, key, column_order, null_precedence, ctx.stream(), ctx.mr());

    if (ctx.nranks == 1) {
      output.move_into(sorted_table->release());
      return;
    }

    auto split_indices = gpu::find_splits_for_distribution(
      ctx, sorted_table->view(), keys_idx, column_order, null_precedence);

    // If the local table has zero rows we cannot split it for sharing and
    // split_indices will be null.  Exchange the (empty) table instead.
    std::vector<cudf::table_view> partitions;
    if (split_indices.size() > 0) {
      partitions = cudf::split(sorted_table->view(), split_indices, ctx.stream());
    } else {
      assert(sorted_table->num_rows() == 0);
      for (int i = 0; i < ctx.nranks; i++) {
        partitions.push_back(sorted_table->view());
      }
    }
    auto [parts, owners] = shuffle(ctx, partitions, std::move(sorted_table));

    std::unique_ptr<cudf::table> result;
    if (!stable) {
      result = cudf::merge(parts, keys_idx, column_order, null_precedence, ctx.stream(), ctx.mr());
    } else {
      // This is not good, but libcudf has no stable merge:
      // https://github.com/rapidsai/cudf/issues/16010
      // https://github.com/rapidsai/cudf/issues/7379
      result = cudf::concatenate(parts, ctx.stream(), ctx.mr());
      owners.reset();  // we created a copy.
      auto res_view = result->view();
      result        = sort_func(
        res_view, res_view.select(keys_idx), column_order, null_precedence, ctx.stream(), ctx.mr());
    }

#if DEBUG_SPLITS
    std::ostringstream result_size_oss;
    result_size_oss << "Rank/chunk " << ctx.rank << " includes " << result->num_rows()
                    << " rows.\n";
    result_size_oss << "    from individual chunks: ";
    for (auto part : parts) {
      result_size_oss << part.num_rows() << ", ";
    }
    std::cout << result_size_oss.str() << std::endl;
#endif
    output.move_into(std::move(result));
  }
};

}  // namespace task

LogicalTable sort(const LogicalTable& tbl,
                  const std::vector<std::string>& keys,
                  const std::vector<bool>& sort_ascending,
                  bool nulls_at_end,
                  bool stable)
{
  if (keys.size() == 0) { throw std::invalid_argument("must sort along at least one column"); }
  if (sort_ascending.size() != keys.size()) {
    throw std::invalid_argument("sort column order and null precedence must match number of keys");
  }

  auto runtime = legate::Runtime::get_runtime();

  auto ret = LogicalTable::empty_like(tbl);

  std::vector<cudf::size_type> keys_idx(keys.size());

  bool use_arrow          = runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0;
  const auto& name_to_idx = tbl.get_column_names();
  for (size_t i = 0; i < keys.size(); i++) {
    if (name_to_idx.count(keys[i]) == 0) {
      throw std::invalid_argument("sort key '" + keys[i] + "' not found in table");
    }
    keys_idx[i] = name_to_idx.at(keys[i]);
  }

  legate::AutoTask task =
    runtime->create_task(get_library(), task::SortTask::TASK_CONFIG.task_id());
  argument::add_next_input(task, tbl);
  argument::add_next_scalar_vector(task, keys_idx);
  argument::add_next_scalar_vector(task, sort_ascending);
  argument::add_next_scalar(task, nulls_at_end);
  argument::add_next_scalar(task, stable);
  argument::add_next_output(task, ret);

  if (use_arrow) {
    task.add_communicator("cpu");
  } else {
    task.add_communicator("nccl");
  }

  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::SortTask::register_variants();
  return 0;
}();

}  // namespace
