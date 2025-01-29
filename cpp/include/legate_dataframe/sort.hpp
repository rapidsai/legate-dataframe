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

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Sort a logical table.
 *
 * Reorder the logical table so that the keys columns are sorted lexicographic
 * based on their column_order and null_precedence.
 *
 * @param tbl The table to sort
 * @param keys The column names to sort by.
 * @param column_order Either ASCENDING or DESCENDING for each sort key/column.
 * @param null_recedence Either BEFORE or AFTER for each sort key/column.
 * AFTER means that nulls are considered larger and come last after an ascending
 * and first after a descending sort.
 * @return The sorted LogicalTable
 */
LogicalTable sort(const LogicalTable& tbl,
                  const std::vector<std::string>& keys,
                  const std::vector<cudf::order>& column_order,
                  const std::vector<cudf::null_order>& null_precedence,
                  bool stable = false);

}  // namespace legate::dataframe
