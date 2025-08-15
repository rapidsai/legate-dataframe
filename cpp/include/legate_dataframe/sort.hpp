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

#pragma once

#include <string>
#include <vector>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {
namespace task {

/**
 * @brief Return points at which to split a dataset.
 *
 * @param nvalues The total number of values to split.
 * @param nsplits the number of splits (and split values as last is included)
 * @param include_start Whether to include the starting 0.
 * @returns column selecting containing nsplits indices.
 */
std::vector<std::size_t> get_split_ind(TaskContext& ctx,
                                       std::size_t nvalues,
                                       int nsplits,
                                       bool include_start);
class SortTask : public Task<SortTask, OpCode::Sort> {
 public:
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(true)
                                                .with_elide_device_ctx_sync(true);
  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);

  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};
}  // namespace task

/**
 * @brief Sort a logical table.
 *
 * Reorder the logical table so that the keys columns are sorted lexicographic
 * based on their column_order and null_precedence. The GPU and CPU backends may not sort NaN values
 * exactly the same way (e.g. according to null_precendence or by treating them as large floating
 * point numbers) - it is recommended to instead use nulls instead of NaNs to get a consistent
 * behaviour between CPU/GPU launches.
 *
 * @param tbl The table to sort
 * @param keys The column names to sort by.
 * @param sort_ascending Whether to sort ascending or descending for each key.
 * @param nulls_at_end Whether nulls are placed at the begging or end (regardless of ascending or
 * descending sort).
 * @param limit The maximum number of rows to return. If negative, the last rows
 * are returns (i.e. a head/tail operation).
 * @return The sorted LogicalTable
 */
LogicalTable sort(const LogicalTable& tbl,
                  const std::vector<std::string>& keys,
                  const std::vector<bool>& sort_ascending,
                  bool nulls_at_end,
                  bool stable                  = false,
                  std::optional<int64_t> limit = std::nullopt);

}  // namespace legate::dataframe
