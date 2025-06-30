/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <utility>
#include <vector>

#include <legate.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/join.hpp>

namespace legate::dataframe {
namespace task {
namespace {

/**
 * @brief Help function to perform a cudf join operation.
 *
 * Since cudf's public join API doesn't accept a stream argument, we use the detail API.
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
cudf_join(TaskContext& ctx,
          cudf::table_view lhs,
          cudf::table_view rhs,
          const std::vector<int32_t>& lhs_keys,
          const std::vector<int32_t>& rhs_keys,
          cudf::null_equality null_equality,
          JoinType join_type)
{
  cudf::hash_join joiner(rhs.select(rhs_keys), null_equality, ctx.stream());

  switch (join_type) {
    case JoinType::INNER: {
      return joiner.inner_join(
        lhs.select(lhs_keys), std::optional<std::size_t>{}, ctx.stream(), ctx.mr());
    }
    case JoinType::LEFT: {
      return joiner.left_join(
        lhs.select(lhs_keys), std::optional<std::size_t>{}, ctx.stream(), ctx.mr());
    }
    case JoinType::FULL: {
      return joiner.full_join(
        lhs.select(lhs_keys), std::optional<std::size_t>{}, ctx.stream(), ctx.mr());
    }
    default: {
      throw std::invalid_argument("Unknown JoinType");
    }
  }
}

/**
 * @brief Help function to get the left and right out-of-bounds-policy for the specified join type
 */
std::pair<cudf::out_of_bounds_policy, cudf::out_of_bounds_policy> out_of_bounds_policy_by_join_type(
  JoinType join_type)
{
  switch (join_type) {
    case JoinType::INNER: {
      return std::make_pair(cudf::out_of_bounds_policy::DONT_CHECK,
                            cudf::out_of_bounds_policy::DONT_CHECK);
    }
    case JoinType::LEFT: {
      return std::make_pair(cudf::out_of_bounds_policy::DONT_CHECK,
                            cudf::out_of_bounds_policy::NULLIFY);
    }
    case JoinType::FULL: {
      return std::make_pair(cudf::out_of_bounds_policy::NULLIFY,
                            cudf::out_of_bounds_policy::NULLIFY);
    }
    default: {
      throw std::invalid_argument("Unknown JoinType");
    }
  }
}

/**
 * @brief Help function to perform a cudf join and gather operation.
 *
 * The result is written to the physical table output
 *
 * Note that `lhs_table` is only passed for cleanup.
 */
void cudf_join_and_gather(TaskContext& ctx,
                          cudf::table_view lhs,
                          cudf::table_view rhs,
                          const std::vector<int32_t> lhs_keys,
                          const std::vector<int32_t> rhs_keys,
                          JoinType join_type,
                          cudf::null_equality null_equality,
                          const std::vector<int32_t> lhs_out_cols,
                          const std::vector<int32_t> rhs_out_cols,
                          PhysicalTable& output,
                          std::unique_ptr<cudf::table> lhs_table = std::unique_ptr<cudf::table>())
{
  // Perform the join and convert (zero-copy) the resulting indices to columns
  auto [lhs_row_idx, rhs_row_idx] =
    cudf_join(ctx, lhs, rhs, lhs_keys, rhs_keys, null_equality, join_type);
  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*lhs_row_idx};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*rhs_row_idx};
  auto left_indices_col   = cudf::column_view{left_indices_span};
  auto right_indices_col  = cudf::column_view{right_indices_span};

  // Use the index columns to gather the result from the original left and right input columns
  auto [left_policy, right_policy] = out_of_bounds_policy_by_join_type(join_type);

  auto left_result =
    cudf::gather(lhs.select(lhs_out_cols), left_indices_col, left_policy, ctx.stream(), ctx.mr());
  // Clean up left indices and columns as quickly as possible to reduce peak memory.
  // (This is the only reason for passing `lhs_table`.)
  lhs_row_idx.reset();
  lhs_table.reset();

  auto right_result =
    cudf::gather(rhs.select(rhs_out_cols), right_indices_col, right_policy, ctx.stream(), ctx.mr());

  // Finally, create a vector of both the left and right results and move it into the output table
  output.move_into(concat(left_result->release(), right_result->release()));
}

/**
 * @brief Help function to create an empty cudf table with no rows
 */
std::unique_ptr<cudf::table> no_rows_table_like(const PhysicalTable& other)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (const auto& dtype : other.cudf_types()) {
    columns.emplace_back(cudf::make_empty_column(dtype));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Help function to "revert" a broadcasted table
 *
 * The table is passed through on rank 0 and on the other ranks, an empty table is returned.
 * The `owners` argument is used to keep new cudf allocations alive
 */
cudf::table_view revert_broadcast(TaskContext& ctx,
                                  const PhysicalTable& table,
                                  std::vector<std::unique_ptr<cudf::table>>& owners)
{
  if (ctx.rank == 0 || table.is_partitioned()) {
    return table.table_view();
  } else {
    owners.push_back(no_rows_table_like(table));
    return owners.back()->view();
  }
}

/**
 * @brief Help function to determine if we need to repartition the tables
 *
 * If legate broadcast the left- or right-hand side table, we might not need to
 * repartition them. This depends on the join type and which table is broadcasted.
 */
bool is_repartition_not_needed(const TaskContext& ctx,
                               JoinType join_type,
                               bool lhs_broadcasted,
                               bool rhs_broadcasted)
{
  if (ctx.nranks == 1) {
    return true;
  } else if (join_type == JoinType::INNER && (lhs_broadcasted || rhs_broadcasted)) {
    return true;
  } else if (join_type == JoinType::LEFT && rhs_broadcasted) {
    return true;
  } else {
    if (ctx.get_legate_context().communicators().size() == 0) {
      throw std::runtime_error(
        "internal join error: repartitioning needed but communicator not set up.");
    }
    return false;
  }
}

}  // namespace

class JoinTask : public Task<JoinTask, OpCode::Join> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::Join}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(true)
                                                .with_elide_device_ctx_sync(true);

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const auto lhs          = argument::get_next_input<PhysicalTable>(ctx);
    const auto rhs          = argument::get_next_input<PhysicalTable>(ctx);
    const auto lhs_keys     = argument::get_next_scalar_vector<int32_t>(ctx);
    const auto rhs_keys     = argument::get_next_scalar_vector<int32_t>(ctx);
    auto join_type          = argument::get_next_scalar<JoinType>(ctx);
    auto null_equality      = argument::get_next_scalar<cudf::null_equality>(ctx);
    const auto lhs_out_cols = argument::get_next_scalar_vector<int32_t>(ctx);
    const auto rhs_out_cols = argument::get_next_scalar_vector<int32_t>(ctx);
    auto output             = argument::get_next_output<PhysicalTable>(ctx);

    /* Use "is_paritioned" to check if the table is broadcast. */
    const bool lhs_broadcasted = !lhs.is_partitioned();
    const bool rhs_broadcasted = !rhs.is_partitioned();
    if (lhs_broadcasted && rhs_broadcasted && ctx.nranks != 1) {
      throw std::runtime_error("join(): cannot have both the lhs and the rhs broadcasted");
    }

    if (is_repartition_not_needed(ctx, join_type, lhs_broadcasted, rhs_broadcasted)) {
      cudf_join_and_gather(ctx,

                           lhs.table_view(),
                           rhs.table_view(),
                           lhs_keys,
                           rhs_keys,
                           join_type,
                           null_equality,
                           lhs_out_cols,
                           rhs_out_cols,
                           output);
    } else {
      std::vector<std::unique_ptr<cudf::table>> owners;

      // All-to-all repartition to one hash bucket per rank. Matching rows from
      // both tables then guaranteed to be on the same rank.
      auto cudf_lhs = repartition_by_hash(ctx, revert_broadcast(ctx, lhs, owners), lhs_keys);
      auto cudf_rhs = repartition_by_hash(ctx, revert_broadcast(ctx, rhs, owners), rhs_keys);

      auto lhs_view = cudf_lhs->view();  // cudf_lhs unique pointer is moved.
      cudf_join_and_gather(ctx,

                           lhs_view,
                           cudf_rhs->view(),
                           lhs_keys,
                           rhs_keys,
                           join_type,
                           null_equality,
                           lhs_out_cols,
                           rhs_out_cols,
                           output,
                           std::move(cudf_lhs)  // to allow early cleanup
      );
    }
  }
};

}  // namespace task

namespace {
/**
 * @brief Help function to append empty columns like those in `table`.
 */
void append_empty_like_columns(std::vector<LogicalColumn>& output, const LogicalTable& table)
{
  for (const auto& col : table.get_columns()) {
    output.push_back(LogicalColumn::empty_like(col));
  }
}

/**
 * @brief Help function to append empty columns like those in `table`.
 * The empty columns are all nullable no matter the nullability of the columns in `table`
 */
void append_empty_like_columns_force_nullable(std::vector<LogicalColumn>& output,
                                              const LogicalTable& table)
{
  for (const auto& col : table.get_columns()) {
    output.push_back(LogicalColumn::empty_like(col.type(), true));
  }
}
}  // namespace

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  const std::vector<size_t>& lhs_out_columns,
                  const std::vector<size_t>& rhs_out_columns,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  auto runtime = legate::Runtime::get_runtime();
  if (lhs_keys.size() != rhs_keys.size()) {
    throw std::invalid_argument("The size of `lhs_keys` and `rhs_keys` must be equal");
  }
  auto lhs_out = lhs.select(lhs_out_columns);
  auto rhs_out = rhs.select(rhs_out_columns);

  // Create an empty like table of the output columns
  std::vector<LogicalColumn> ret_cols;
  switch (join_type) {
    case JoinType::INNER: {
      append_empty_like_columns(ret_cols, lhs_out);
      append_empty_like_columns(ret_cols, rhs_out);
      break;
    }
    case JoinType::LEFT: {
      append_empty_like_columns(ret_cols, lhs_out);
      // In a left join, the right columns might contain nulls even when `rhs` doesn't
      append_empty_like_columns_force_nullable(ret_cols, rhs_out);
      break;
    }
    case JoinType::FULL: {
      // In a full join, both left and right columns might contain nulls
      // even when `lhs` or `rhs` doesn't
      append_empty_like_columns_force_nullable(ret_cols, lhs_out);
      append_empty_like_columns_force_nullable(ret_cols, rhs_out);
      break;
    }
    default: {
      throw std::invalid_argument("Unknown JoinType");
    }
  }

  // Create the output table
  auto ret_names = concat(lhs_out.get_column_name_vector(), rhs_out.get_column_name_vector());
  auto ret       = LogicalTable(std::move(ret_cols), std::move(ret_names));

  legate::AutoTask task =
    runtime->create_task(get_library(), task::JoinTask::TASK_CONFIG.task_id());
  // TODO: While legate may broadcast some arrays, it would be good to add
  //       a heuristic (e.g. based on the fact that we need to do copies
  //       anyway, so the broadcast may actually copy less).
  //       That could be done here, in a mapper, or within the task itself.
  argument::add_next_input(task, lhs, broadcast == BroadcastInput::LEFT);
  argument::add_next_input(task, rhs, broadcast == BroadcastInput::RIGHT);
  argument::add_next_scalar_vector(task, std::vector<int32_t>(lhs_keys.begin(), lhs_keys.end()));
  argument::add_next_scalar_vector(task, std::vector<int32_t>(rhs_keys.begin(), rhs_keys.end()));
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<JoinType>>(join_type));
  argument::add_next_scalar(
    task, static_cast<std::underlying_type_t<cudf::null_equality>>(compare_nulls));
  argument::add_next_scalar_vector(
    task, std::vector<int32_t>(lhs_out_columns.begin(), lhs_out_columns.end()));
  argument::add_next_scalar_vector(
    task, std::vector<int32_t>(rhs_out_columns.begin(), rhs_out_columns.end()));
  argument::add_next_output(task, ret);
  if (broadcast == BroadcastInput::AUTO) {
    task.add_communicator("nccl");
  } else if (join_type == JoinType::FULL ||
             (broadcast == BroadcastInput::LEFT && join_type != JoinType::INNER)) {
    throw std::runtime_error(
      "Force broadcast was indicated, but repartitioning is required. "
      "FULL joins do not support broadcasting and LEFT joins only for the "
      "right hand side argument.");
  }
  runtime->submit(std::move(task));
  return ret;
}

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<std::string>& lhs_keys,
                  const std::set<std::string>& rhs_keys,
                  JoinType join_type,
                  const std::vector<std::string>& lhs_out_columns,
                  const std::vector<std::string>& rhs_out_columns,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  // Convert column names to indices
  std::set<size_t> lhs_keys_idx;
  std::set<size_t> rhs_keys_idx;
  std::vector<size_t> lhs_out_columns_idx;
  std::vector<size_t> rhs_out_columns_idx;
  const auto& lhs_name_to_idx = lhs.get_column_names();
  const auto& rhs_name_to_idx = rhs.get_column_names();
  for (const auto& name : lhs_keys) {
    lhs_keys_idx.insert(lhs_name_to_idx.at(name));
  }
  for (const auto& name : rhs_keys) {
    rhs_keys_idx.insert(rhs_name_to_idx.at(name));
  }
  for (const auto& name : lhs_out_columns) {
    lhs_out_columns_idx.push_back(lhs_name_to_idx.at(name));
  }
  for (const auto& name : rhs_out_columns) {
    rhs_out_columns_idx.push_back(rhs_name_to_idx.at(name));
  }
  return join(lhs,
              rhs,
              lhs_keys_idx,
              rhs_keys_idx,
              join_type,
              lhs_out_columns_idx,
              rhs_out_columns_idx,
              compare_nulls,
              broadcast);
}

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  // By default, the output includes all the columns from `lhs` and `rhs`.
  std::vector<size_t> lhs_out_columns(lhs.num_columns());
  std::iota(lhs_out_columns.begin(), lhs_out_columns.end(), 0);
  std::vector<size_t> rhs_out_columns(rhs.num_columns());
  std::iota(rhs_out_columns.begin(), rhs_out_columns.end(), 0);
  return join(lhs,
              rhs,
              lhs_keys,
              rhs_keys,
              join_type,
              lhs_out_columns,
              rhs_out_columns,
              compare_nulls,
              broadcast);
}

}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::JoinTask::register_variants();
  return 0;
}();

}  // namespace
