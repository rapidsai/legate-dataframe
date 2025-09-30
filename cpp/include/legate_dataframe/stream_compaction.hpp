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

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {
namespace task {
class ApplyBooleanMaskTask : public Task<ApplyBooleanMaskTask, OpCode::ApplyBooleanMask> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

class DistinctTask : public Task<DistinctTask, OpCode::Distinct> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_concurrent(true)
                                                .with_elide_device_ctx_sync(true);
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace task

/**
 * @brief Filter a table busing a boolean mask.
 *
 * Select all rows from the table where the boolean mask column is true
 * (non-null and not false).  The operation is stable.
 *
 * @param tbl The table to filter.
 * @param boolean_mask The boolean mask to apply.
 * @return The LogicalTable containing only the rows where the boolean_mask was true.
 */
LogicalTable apply_boolean_mask(const LogicalTable& tbl, const LogicalColumn& boolean_mask);

/**
 * @brief Find distinct (unique) rows in a logical table.
 *
 * For rows not listed as keys any value is possible, order is not guaranteed.
 *
 * @param tbl The table to distinct.
 * @param keys The column names to distinct by.
 * @param high_cardinality Whether the table is assumed to have a high cardinality.
 * If false (default), the cardinality is assumed to be low in which case we can save
 * communication by doing a local distinct before shuffling data.  If the cardinality is high,
 * this would double the work, however.
 * @return The LogicalTable containing only the distinct rows.
 */
LogicalTable distinct(const LogicalTable& tbl,
                      const std::vector<std::string>& keys,
                      bool high_cardinality = false);

}  // namespace legate::dataframe
