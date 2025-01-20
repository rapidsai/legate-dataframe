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

#include <legate.h>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

std::pair<std::vector<cudf::table_view>,
          std::unique_ptr<std::pair<std::map<int, rmm::device_buffer>, cudf::table>>>
shuffle(GPUTaskContext& ctx,
        std::vector<cudf::table_view>& tbl_partitioned,
        std::unique_ptr<cudf::table> owning_table);

/**
 * @brief Repartition the table into hash table buckets for each rank/node.
 *
 * After partitioning, each rank owns the data for one hash bucket based on
 * the given columns.  Rows matching in these columns are guaranteed to be
 * in the same bucket.
 *
 * This assumes that the calling task was created with a NCCL communicator,
 * see `legate::AutoTask::add_communicator()`.
 *
 * @param ctx The context of the calling task
 * @param table The table to repartition
 * @param columns_to_hash Indices of input columns to hash
 * @return The repartitioned table where the partition of each task hashes to the same
 */
std::unique_ptr<cudf::table> repartition_by_hash(
  GPUTaskContext& ctx,
  const cudf::table_view& table,
  const std::vector<cudf::size_type>& columns_to_hash);

}  // namespace legate::dataframe::task
