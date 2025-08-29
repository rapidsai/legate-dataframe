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

#include <legate.h>
#include <legate/cuda/cuda.h>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/join.hpp>
#include <legate_dataframe/sort.hpp>

#define DEBUG_SPLITS 0
#if DEBUG_SPLITS
#include <iostream>
#include <sstream>
#endif

namespace legate::dataframe {
namespace task {

std::vector<std::size_t> get_split_ind(TaskContext& ctx,
                                       std::size_t nvalues,
                                       int nsplits,
                                       bool include_start)
{
  std::size_t nvalues_per_split, nvalues_left;
  if (nvalues < nsplits) {
    // Only return non-empty splits (we could return one point multiple times)
    nsplits           = nvalues;
    nvalues_per_split = (nsplits > 0);
    nvalues_left      = 0;
  } else {
    nvalues_per_split = nvalues / nsplits;
    nvalues_left      = nvalues - nvalues_per_split * nsplits;
  }

  std::vector<std::size_t> split_values;
  std::size_t split_offset = 0;

  if (include_start && nvalues > 0) { split_values.push_back(0); }

  // nsplits could be 0
  for (std::int64_t i = 0; i < nsplits - 1; i++) {
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

namespace {

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
  auto status = builder.Reserve(num_elements);
  for (int64_t i = 0; i < num_elements; i++) {
    builder.UnsafeAppend(fill_value);
  }
  return ARROW_RESULT(builder.Finish());
}

std::shared_ptr<arrow::Table> extract_local_splits(TaskContext& ctx,
                                                   std::shared_ptr<arrow::Table> sorted_table,
                                                   const std::vector<std::size_t>& keys_idx)
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
  const std::vector<std::size_t>& keys_idx,
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
// Compare two tables each containing a single row
bool compare(TaskContext& ctx,
             std::shared_ptr<arrow::Table> row_a,
             std::shared_ptr<arrow::Table> row_b,
             const std::vector<std::size_t>& keys_idx,
             const std::vector<arrow::compute::SortKey>& column_order,
             arrow::compute::NullPlacement null_precedence)
{
  // Create a table with each row, sort that table
  // Use the resulting order as the comparison
  // We are testing row_a < row_b
  // So if row_a gets sorted into position 0, then row_a < row_b

  auto combined = ARROW_RESULT(arrow::ConcatenateTables({row_b, row_a}));

  auto sort_indices = ARROW_RESULT(arrow::compute::SortIndices(
    combined, arrow::compute::SortOptions(column_order, null_precedence)));

  std::shared_ptr<arrow::UInt64Scalar> first_sort_index =
    std::dynamic_pointer_cast<arrow::UInt64Scalar>(ARROW_RESULT(sort_indices->GetScalar(0)));
  return first_sort_index->value == 1;
}

std::size_t lower_bound_row(TaskContext& ctx,
                            std::shared_ptr<arrow::Table> haystack,
                            std::shared_ptr<arrow::Table> needle,  // This is a single row
                            const std::vector<std::size_t>& keys_idx,
                            const std::vector<arrow::compute::SortKey>& column_order,
                            arrow::compute::NullPlacement null_precedence)
{
  std::size_t first  = 0;
  std::size_t length = haystack->num_rows();
  while (length > 0) {
    auto rem = length % 2;
    length /= 2;
    if (compare(ctx,
                haystack->Slice(first + length, 1),
                needle,
                keys_idx,
                column_order,
                null_precedence)) {
      first += length + rem;
    }
  }
  return first;
}

std::vector<std::size_t> find_destination_ranks(
  TaskContext& ctx,
  std::shared_ptr<arrow::Table> sorted_table,
  std::shared_ptr<arrow::Table> global_split_values,
  const std::vector<std::size_t>& keys_idx,
  const std::vector<arrow::compute::SortKey>& column_order,
  arrow::compute::NullPlacement null_precedence)
{
  // Create a new table with the rank column appended
  auto rank_array = create_array<int64_t>(sorted_table->num_rows(), ctx.rank);
  auto sorted_table_with_rank =
    ARROW_RESULT(sorted_table->AddColumn(sorted_table->num_columns(),
                                         arrow::field("split_rank", rank_array->type()),
                                         std::make_shared<arrow::ChunkedArray>(rank_array)));

  std::vector<std::size_t> splits_indices_host;

  auto keys_idx_with_rank = keys_idx;
  keys_idx_with_rank.push_back(sorted_table->num_columns());
  auto column_order_with_rank = column_order;
  column_order_with_rank.push_back(
    arrow::compute::SortKey{"split_rank", arrow::compute::SortOrder::Ascending});

  // For each global split value, find where it should be inserted in the local sorted table
  for (std::size_t i = 0; i < static_cast<std::size_t>(global_split_values->num_rows()); i++) {
    auto split_value = global_split_values->Slice(i, 1);
    // Remove the index column so the columns in the comparison are the same
    split_value          = ARROW_RESULT(split_value->RemoveColumn(split_value->num_columns() - 1));
    std::size_t position = lower_bound_row(ctx,
                                           sorted_table_with_rank,
                                           split_value,
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
  const std::vector<std::size_t>& keys_idx,
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

static std::shared_ptr<arrow::Table> apply_limit(std::shared_ptr<arrow::Table> tbl, int64_t limit)
{
  if (limit != INT64_MIN && std::abs(limit) < tbl->num_rows()) {
    std::shared_ptr<arrow::Table> slice;
    if (limit < 0) {
      slice = tbl->Slice(tbl->num_rows() + limit, -limit);
    } else {
      slice = tbl->Slice(0, limit);
    }
    return slice;
  }
  return tbl;
}

}  // namespace

/*static*/ void SortTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto tbl            = argument::get_next_input<PhysicalTable>(ctx);
  const auto keys_idx       = argument::get_next_scalar_vector<std::size_t>(ctx);
  const auto sort_ascending = argument::get_next_scalar_vector<bool>(ctx);
  const auto nulls_at_end   = argument::get_next_scalar<bool>(ctx);
  const auto stable         = argument::get_next_scalar<bool>(ctx);
  const auto limit          = argument::get_next_scalar<int64_t>(ctx);
  auto output               = argument::get_next_output<PhysicalTable>(ctx);

  // Sort the table
  std::vector<arrow::compute::SortKey> sort_keys;
  for (size_t i = 0; i < keys_idx.size(); i++) {
    // Translate cudf parameters to arrow parameters
    auto order = sort_ascending[i] ? arrow::compute::SortOrder::Ascending
                                   : arrow::compute::SortOrder::Descending;
    sort_keys.push_back(arrow::compute::SortKey{std::to_string(keys_idx[i]), order});
  }
  auto arrow_table = tbl.arrow_table_view();
  // Arrow does not support null_order per column, so we use the first one
  auto null_order =
    nulls_at_end ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart;
  arrow::compute::SortOptions sort_options(sort_keys, null_order);
  auto sorted_indices = ARROW_RESULT(arrow::compute::SortIndices(arrow_table, sort_options));
  auto sorted_table =
    ARROW_RESULT(arrow::compute::Take(arrow_table, *sorted_indices, arrow::compute::TakeOptions{}))
      .table();

  sorted_table = apply_limit(sorted_table, limit);

  if (ctx.nranks == 1) {
    output.move_into(sorted_table);
    return;
  }

  auto split_indices =
    find_splits_for_distribution(ctx, sorted_table, keys_idx, sort_keys, null_order);

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
}  // namespace task

LogicalTable sort(const LogicalTable& tbl,
                  const std::vector<std::string>& keys,
                  const std::vector<bool>& sort_ascending,
                  bool nulls_at_end,
                  bool stable,
                  std::optional<int64_t> limit)
{
  if (keys.size() == 0) { throw std::invalid_argument("must sort along at least one column"); }
  if (sort_ascending.size() != keys.size()) {
    throw std::invalid_argument("sort column order and null precedence must match number of keys");
  }

  auto runtime = legate::Runtime::get_runtime();

  auto ret = LogicalTable::empty_like(tbl);

  std::vector<std::size_t> keys_idx(keys.size());

  bool use_arrow          = runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0;
  const auto& name_to_idx = tbl.get_column_names();
  auto keys_set           = std::unordered_set<std::string>(keys.begin(), keys.end());
  if (keys_set.size() != keys.size()) { throw std::invalid_argument("duplicate sort keys"); }
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
  argument::add_next_scalar(task, limit.has_value() ? limit.value() : INT64_MIN);
  argument::add_next_output(task, ret);

  if (use_arrow) {
    task.add_communicator("cpu");
  } else {
    task.add_communicator("nccl");
  }

  runtime->submit(std::move(task));
  if (limit.has_value()) {
    if (limit.value() < 0) {
      ret = ret.slice({limit.value(), legate::Slice::OPEN});
    } else {
      ret = ret.slice({0, limit.value()});
    }
  }
  return ret;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::SortTask::register_variants();
}

}  // namespace
