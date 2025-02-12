/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>  // cudf::null_equality

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

enum class JoinType : int32_t { INNER = 0, LEFT, FULL };
enum class BroadcastInput : int32_t { AUTO = 0, LEFT, RIGHT };

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
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @param _num_partitions TODO: For testing.
 * The  number of partitions to use.  If -1 uses the nccl approach instead.  If
 * broadcast is not AUTO, this is ignored (no shuffling is needed).
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  const std::vector<size_t>& lhs_out_columns,
                  const std::vector<size_t>& rhs_out_columns,
                  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                  BroadcastInput broadcast          = BroadcastInput::AUTO,
                  int _num_partitions = -1);

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
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @param _num_partitions TODO: For testing.
 * The  number of partitions to use.  If -1 uses the nccl approach instead.  If
 * broadcast is not AUTO, this is ignored (no shuffling is needed).
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<std::string>& lhs_keys,
                  const std::set<std::string>& rhs_keys,
                  JoinType join_type,
                  const std::vector<std::string>& lhs_out_columns,
                  const std::vector<std::string>& rhs_out_columns,
                  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                  BroadcastInput broadcast          = BroadcastInput::AUTO,
                  int _num_partitions = -1);

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
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param broadcast Which, if any, of the inputs should be copied to all workers.
 * @return The joining result
 */
LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                  BroadcastInput broadcast          = BroadcastInput::AUTO);

}  // namespace legate::dataframe
