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

#include <legate.h>
#include <legate/cuda/cuda.h>

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
std::unique_ptr<cudf::column> get_split_ind(TaskContext& ctx,
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

  auto ncopy = split_values.size();
  rmm::device_uvector<cudf::size_type> split_ind(ncopy, ctx.stream(), ctx.mr());
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_ind.data(),
                                    split_values.data(),
                                    ncopy * sizeof(cudf::size_type),
                                    cudaMemcpyHostToDevice,
                                    ctx.stream()));
  LEGATE_CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));

  return std::make_unique<cudf::column>(std::move(split_ind), std::move(rmm::device_buffer()), 0);
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
std::unique_ptr<std::vector<cudf::size_type>> find_splits_for_distribution(
  TaskContext& ctx,
  const cudf::table_view& my_sorted_tbl,
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
  auto my_split_ind_col =
    get_split_ind(ctx, my_sorted_tbl.num_rows(), ctx.nranks, /* include_start */ true);
  auto nsplits = my_split_ind_col->size();

  auto my_split_rank_col = cudf::sequence(nsplits,
                                          *cudf::make_fixed_width_scalar(int32_t{ctx.rank}),
                                          *cudf::make_fixed_width_scalar(int32_t{0}));

  auto my_split_cols_tbl = cudf::gather(my_sorted_tbl.select(keys_idx),
                                        my_split_ind_col->view(),
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        ctx.stream());

  auto my_split_cols_view = my_split_cols_tbl->view();
  auto my_split_cols_vector =
    std::vector<cudf::column_view>(my_split_cols_view.begin(), my_split_cols_view.end());

  // Add in rank and local index (together provide a global order).
  my_split_cols_vector.push_back(my_split_rank_col->view());
  my_split_cols_vector.push_back(my_split_ind_col->view());
  auto my_splits = cudf::table_view(my_split_cols_vector);

  // keys(x) to pick columns from splits (which include rank and index):
  std::vector<cudf::size_type> value_keysx(keys_idx.size());
  std::iota(value_keysx.begin(), value_keysx.end(), 0);
  std::vector<cudf::size_type> all_keysx(keys_idx.size() + 2);
  std::iota(all_keysx.begin(), all_keysx.end(), 0);

  /*
   * Step 2: Share split candidates among all ranks.
   */
  std::vector<cudf::table_view> exchange_tables;
  for (int i = 0; i < ctx.nranks; i++) {
    exchange_tables.push_back(my_splits);
  }
  auto [split_candidates_shared, owners_split] = shuffle(ctx, exchange_tables, nullptr);
  if (my_split_cols_tbl->num_rows() == 0) {
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
  auto split_candidates = cudf::merge(
    split_candidates_shared, all_keysx, column_orderx, null_precedencex, ctx.stream(), ctx.mr());
  owners_split.reset();  // copied into split_candidates

  /*
   * Step 3: Find the best splitting points from all candidates
   */
  auto split_value_inds =
    get_split_ind(ctx, split_candidates->num_rows(), ctx.nranks, /* include_start */ false);
  auto split_values_tbl  = cudf::gather(split_candidates->view(),
                                       split_value_inds->view(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       ctx.stream());
  auto split_values_view = split_values_tbl->view();

  /*
   * Step 4: Find the actual split points for the local dataset.
   *
   * We need to split based on the rank of the split point `split_rank`
   * (i.e. where is the split point in the whole dataset):
   *    - if split_rank < my_rank:  split at first equal row.
   *    - if split_rank == my_rank:  use split-point index.
   *    - if split_rank > my_rank: split after last equal row
   *
   * N.B.: If this turns out to matter speed-wise, this can be spelled as a single
   * `lower_bound` with the (global) row-index.  A custom implementation could
   * make that row-index a virtual table.
   */
  auto split_candidates_first_col  = cudf::lower_bound(my_sorted_tbl.select(keys_idx),
                                                      split_values_view.select(value_keysx),
                                                      column_order,
                                                      null_precedence,
                                                      ctx.stream());
  auto split_candidates_first_view = split_candidates_first_col->view();
  auto split_candidates_last_col   = cudf::upper_bound(my_sorted_tbl.select(keys_idx),
                                                     split_values_view.select(value_keysx),
                                                     column_order,
                                                     null_precedence,
                                                     ctx.stream());
  auto split_candidates_last_view  = split_candidates_last_col->view();

  // The local index and rank of the split value, we'll use the rank if it came from this rank
  auto split_candidates_equal_view = split_values_view.column(my_splits.num_columns() - 1);
  auto split_candiates_rank_view   = split_values_view.column(my_splits.num_columns() - 2);

  /*
   * Copy all the above information to the host and finalize the local splits.
   * (we may have fewer than nranks split-points here and need to pad later.)
   */
  auto nsplitpoints = split_value_inds->size();
  std::vector<cudf::size_type> split_candidates_first(nsplitpoints);
  std::vector<cudf::size_type> split_candidates_last(nsplitpoints);
  std::vector<cudf::size_type> split_candidates_equal(nsplitpoints);
  std::vector<int32_t> split_candidates_rank(nsplitpoints);

  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_candidates_first.data(),
                                    split_candidates_first_view.data<cudf::size_type>(),
                                    nsplitpoints * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    ctx.stream()));
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_candidates_last.data(),
                                    split_candidates_last_view.data<cudf::size_type>(),
                                    nsplitpoints * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    ctx.stream()));
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_candidates_equal.data(),
                                    split_candidates_equal_view.data<cudf::size_type>(),
                                    nsplitpoints * sizeof(cudf::size_type),
                                    cudaMemcpyDeviceToHost,
                                    ctx.stream()));
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(split_candidates_rank.data(),
                                    split_candiates_rank_view.data<int32_t>(),
                                    nsplitpoints * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost,
                                    ctx.stream()));

  LEGATE_CHECK_CUDA(cudaStreamSynchronize(ctx.stream()));

  auto splits_host = std::make_unique<std::vector<cudf::size_type>>();
  for (int i = 0; i < nsplitpoints; i++) {
    if (split_candidates_rank[i] < ctx.rank) {
      splits_host->push_back(split_candidates_first[i]);
    } else if (split_candidates_rank[i] > ctx.rank) {
      splits_host->push_back(split_candidates_last[i]);
    } else {
      splits_host->push_back(split_candidates_equal[i]);
    }
  }
  // In the obscure case where there is less data than ranks, pad split points.
  for (int i = nsplitpoints; i < ctx.nranks - 1; i++) {
    splits_host->push_back(my_sorted_tbl.num_rows());
  }

#if DEBUG_SPLITS
  std::ostringstream full_splits_oss;
  full_splits_oss << "Final local split points @" << ctx.rank
                  << " (nrows=" << my_sorted_tbl.num_rows() << "):\n";
  for (int i = 0; i < ctx.nranks - 1; i++) {
    full_splits_oss << "    " << splits_host->at(i);
    if (i < nsplitpoints) {
      full_splits_oss << ", split by r" << split_candidates_rank[i] << ": ";
      full_splits_oss << split_candidates_first[i] << "<" << split_candidates_last[i];
      full_splits_oss << ", r[ind]=" << split_candidates_equal[i] << "\n";
    } else {
      full_splits_oss << " (no further split candidates)\n";
    }
  }
  std::cout << full_splits_oss.str() << std::endl;
#endif
  return std::move(splits_host);
}

}  // namespace

class SortTask : public Task<SortTask, OpCode::Sort> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::Sort}};

  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto tbl             = argument::get_next_input<PhysicalTable>(ctx);
    const auto keys_idx        = argument::get_next_scalar_vector<cudf::size_type>(ctx);
    const auto column_order    = argument::get_next_scalar_vector<cudf::order>(ctx);
    const auto null_precedence = argument::get_next_scalar_vector<cudf::null_order>(ctx);
    const auto stable          = argument::get_next_scalar<bool>(ctx);
    auto output                = argument::get_next_output<PhysicalTable>(ctx);

    // Create a new locally sorted table (we always need this)
    auto cudf_tbl  = tbl.table_view();
    auto key       = cudf_tbl.select(keys_idx);
    auto sort_func = stable ? cudf::stable_sort_by_key : cudf::sort_by_key;
    auto my_sorted_tbl =
      sort_func(cudf_tbl, key, column_order, null_precedence, ctx.stream(), ctx.mr());

    if (ctx.nranks == 1) {
      output.move_into(my_sorted_tbl->release());
      return;
    }

    auto split_indices = find_splits_for_distribution(
      ctx, my_sorted_tbl->view(), keys_idx, column_order, null_precedence);

    // If the local table has zero rows we cannot split it for sharing and
    // split_indices will be null.  Exchange the (empty) table instead.
    std::vector<cudf::table_view> partitions;
    if (split_indices) {
      partitions = cudf::split(my_sorted_tbl->view(), *split_indices, context.get_task_stream());
    } else {
      assert(my_sorted_tbl->num_rows() == 0);
      for (int i = 0; i < ctx.nranks; i++) {
        partitions.push_back(my_sorted_tbl->view());
      }
    }
    auto [parts, owners] = shuffle(ctx, partitions, std::move(my_sorted_tbl));

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
      result        = sort_func(res_view,
                         res_view.select(keys_idx),
                         column_order,
                         null_precedence,
                         context.get_task_stream(),
                         ctx.mr());
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
                  const std::vector<cudf::order>& column_order,
                  const std::vector<cudf::null_order>& null_precedence,
                  bool stable)
{
  if (keys.size() == 0) { throw std::invalid_argument("must sort along at least one column"); }
  if (column_order.size() != keys.size() || null_precedence.size() != keys.size()) {
    throw std::invalid_argument("sort column order and null precedence must match number of keys");
  }

  auto runtime = legate::Runtime::get_runtime();

  auto ret = LogicalTable::empty_like(tbl);

  std::vector<cudf::size_type> keys_idx(keys.size());
  std::vector<std::underlying_type_t<cudf::order>> column_order_lg(keys.size());
  std::vector<std::underlying_type_t<cudf::null_order>> null_precedence_lg(keys.size());

  const auto& name_to_idx = tbl.get_column_names();
  for (size_t i = 0; i < keys.size(); i++) {
    keys_idx[i]        = name_to_idx.at(keys[i]);
    column_order_lg[i] = static_cast<std::underlying_type_t<cudf::order>>(column_order[i]);
    null_precedence_lg[i] =
      static_cast<std::underlying_type_t<cudf::null_order>>(null_precedence[i]);
  }

  legate::AutoTask task =
    runtime->create_task(get_library(), task::SortTask::TASK_CONFIG.task_id());
  argument::add_next_input(task, tbl);
  argument::add_next_scalar_vector(task, keys_idx);
  argument::add_next_scalar_vector(task, column_order_lg);
  argument::add_next_scalar_vector(task, null_precedence_lg);
  argument::add_next_scalar(task, stable);
  argument::add_next_output(task, ret);

  task.add_communicator("nccl");

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
