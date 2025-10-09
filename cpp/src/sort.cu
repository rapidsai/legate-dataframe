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
#include <legate_dataframe/sort.hpp>

#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/join.hpp>
#include <legate_dataframe/utils.hpp>

#define DEBUG_SPLITS 0
#if DEBUG_SPLITS
#include <iostream>
#include <sstream>
#endif

namespace legate::dataframe::task {

namespace {
std::unique_ptr<cudf::column> vector_to_column(const std::vector<std::size_t>& vec,
                                               TaskContext& ctx)
{
  auto ncopy = vec.size();
  rmm::device_uvector<std::size_t> split_ind(ncopy, ctx.stream(), ctx.mr());
  LDF_CUDA_TRY(cudaMemcpyAsync(split_ind.data(),
                               vec.data(),
                               ncopy * sizeof(std::size_t),
                               cudaMemcpyHostToDevice,
                               ctx.stream()));
  LDF_CUDA_TRY(cudaStreamSynchronize(ctx.stream()));

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
                        *cudf::make_fixed_width_scalar(fill_value, ctx.stream(), ctx.mr()),
                        *cudf::make_fixed_width_scalar(int32_t{0}, ctx.stream(), ctx.mr()),
                        ctx.stream(),
                        ctx.mr());
}

template <typename T>
std::vector<T> column_to_vector(TaskContext& ctx, const cudf::column_view& col)
{
  std::vector<T> ret(col.size());
  if (col.size() > 0) {
    LDF_CUDA_TRY(cudaMemcpyAsync(
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

  auto column_order_with_rank = column_order;
  column_order_with_rank.push_back(cudf::order::ASCENDING);
  auto null_precendence_with_rank = null_precedence;
  null_precendence_with_rank.push_back(cudf::null_order::AFTER);
  auto split_indices = cudf::lower_bound(sorted_table_with_rank.select(keys_idxx),
                                         global_split_values.select(value_keysx),
                                         column_order_with_rank,
                                         null_precendence_with_rank,
                                         ctx.stream(),
                                         ctx.mr());

  /*
   * Copy the split candidates to the host and finalize the local splits.
   * (we may have fewer than nranks split-points here and need to pad later.)
   */
  auto splits_indices_host = column_to_vector<cudf::size_type>(ctx, split_indices->view());
  LDF_CUDA_TRY(cudaStreamSynchronize(ctx.stream()));
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

static std::unique_ptr<cudf::table> apply_limit(TaskContext& ctx,
                                                std::unique_ptr<cudf::table> tbl,
                                                int64_t limit)
{
  if (limit != INT64_MIN && std::abs(limit) < tbl->num_rows()) {
    cudf::size_type cudf_limit = static_cast<cudf::size_type>(limit);
    cudf::table_view slice;
    if (limit < 0) {
      slice =
        cudf::slice(tbl->view(), {tbl->num_rows() + cudf_limit, tbl->num_rows()}, ctx.stream())[0];
    } else {
      slice = cudf::slice(tbl->view(), {0, cudf_limit}, ctx.stream())[0];
    }
    tbl = std::make_unique<cudf::table>(slice);
  }
  return tbl;
}

}  // namespace

/*static*/ void SortTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto tbl       = argument::get_next_input<PhysicalTable>(ctx);
  const auto keys_idx_ = argument::get_next_scalar_vector<std::size_t>(ctx);
  std::vector<cudf::size_type> keys_idx(keys_idx_.begin(),
                                        keys_idx_.end());  // Change to cudf size type
  const auto sort_ascending = argument::get_next_scalar_vector<bool>(ctx);
  const auto nulls_at_end   = argument::get_next_scalar<bool>(ctx);
  const auto stable         = argument::get_next_scalar<bool>(ctx);
  const auto limit          = argument::get_next_scalar<int64_t>(ctx);
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
      null_precedence.push_back(nulls_at_end ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
    } else {
      null_precedence.push_back(nulls_at_end ? cudf::null_order::AFTER : cudf::null_order::BEFORE);
    }
  }

  // Create a new locally sorted table (we always need this)
  auto cudf_tbl  = tbl.table_view();
  auto key       = cudf_tbl.select(keys_idx);
  auto sort_func = stable ? cudf::stable_sort_by_key : cudf::sort_by_key;
  auto sorted_table =
    sort_func(cudf_tbl, key, column_order, null_precedence, ctx.stream(), ctx.mr());

  sorted_table = apply_limit(ctx, std::move(sorted_table), limit);

  if (ctx.nranks == 1) {
    output.move_into(sorted_table->release());
    return;
  }

  auto split_indices = find_splits_for_distribution(
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
  result_size_oss << "Rank/chunk " << ctx.rank << " includes " << result->num_rows() << " rows.\n";
  result_size_oss << "    from individual chunks: ";
  for (auto part : parts) {
    result_size_oss << part.num_rows() << ", ";
  }
  std::cout << result_size_oss.str() << std::endl;
#endif
  output.move_into(std::move(result));
}

}  // namespace legate::dataframe::task
