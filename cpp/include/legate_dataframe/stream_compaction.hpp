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
  static void gpu_variant(legate::TaskContext context);
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

}  // namespace legate::dataframe
