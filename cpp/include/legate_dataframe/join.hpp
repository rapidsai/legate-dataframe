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

#pragma once

#include <set>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {
enum class JoinType : int32_t { INNER = 0, LEFT, FULL, SEMI, ANTI };
enum class BroadcastInput : int32_t { AUTO = 0, LEFT, RIGHT };

namespace task {
template <bool needs_communication>
class JoinTask : public Task<JoinTask<needs_communication>,
                             needs_communication ? OpCode::JoinConcurrent : OpCode::Join> {
 public:
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(needs_communication)
                                                .with_elide_device_ctx_sync(true);
#endif
  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(needs_communication);

  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};
/**
 * @brief Help function to determine if we need to repartition the tables
 *
 * If legate broadcast the left- or right-hand side table, we might not need to
 * repartition them. This depends on the join type and which table is broadcasted.
 */
bool is_repartition_not_needed(const TaskContext& ctx,
                               JoinType join_type,
                               bool lhs_broadcasted,
                               bool rhs_broadcasted);
}  // namespace task
/**
 * @brief Perform a join between the specified tables.
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @throw invalid_argument if the column names of `lhs_out_columns` and `rhs_out_columns`
 * are not unique.
 *
 * @param lhs The left table
 * @param rhs The right table
 * @param left_keys The column indices of the left table to join on
 * @param right_keys The column indices of the right table to join on
 * @param join_type The join type such as INNER, LEFT, or FULL
 * @param lhs_out_columns Indices of the left hand table columns to include in the result.
 * @param rhs_out_columns Indices of the right hand table columns to include in the result.
 * @param nulls_equal Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::vector<size_t>& lhs_keys,
                  const std::vector<size_t>& rhs_keys,
                  JoinType join_type,
                  const std::vector<size_t>& lhs_out_columns,
                  const std::vector<size_t>& rhs_out_columns,
                  bool nulls_equal         = true,
                  BroadcastInput broadcast = BroadcastInput::AUTO);

/**
 * @brief Perform a join between the specified tables.
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @throw invalid_argument if the column names of `lhs_out_columns` and `rhs_out_columns`
 * are not unique.
 *
 * @param lhs The left table
 * @param rhs The right table
 * @param left_keys The column names of the left table to join on
 * @param right_keys The column names of the right table to join on
 * @param join_type The join type such as INNER, LEFT, or FULL
 * @param lhs_out_columns Names of the left hand table columns to include in the result.
 * @param rhs_out_columns Names of the right hand table columns to include in the result.
 * @param nulls_equal Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::vector<std::string>& lhs_keys,
                  const std::vector<std::string>& rhs_keys,
                  JoinType join_type,
                  const std::vector<std::string>& lhs_out_columns,
                  const std::vector<std::string>& rhs_out_columns,
                  bool nulls_equal         = true,
                  BroadcastInput broadcast = BroadcastInput::AUTO);

/**
 * @brief Perform a join between the specified tables.
 *
 * The joining result of this overload includes all the columns of `lhs` and 'rhs`.
 * In order to select the desired output columns, please use the `lhs_out_columns` and
 * `rhs_out_columns` arguments in the other overloads.
 *
 * @throw cudf::logic_error if number of elements in `left_keys` or `right_keys`
 * mismatch.
 *
 * @throw invalid_argument if the column names of `lhs` and `rhs` are not unique.
 *
 * @param lhs The left table
 * @param rhs The right table
 * @param left_keys The column indices of the left table to join on
 * @param right_keys The column indices of the right table to join on
 * @param join_type The join type such as INNER, LEFT, or FULL
 * @param nulls_equal Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::vector<size_t>& lhs_keys,
                  const std::vector<size_t>& rhs_keys,
                  JoinType join_type,
                  bool nulls_equal         = true,
                  BroadcastInput broadcast = BroadcastInput::AUTO);

}  // namespace legate::dataframe
