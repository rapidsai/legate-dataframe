/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Perform a groupby and aggregation in a single operation.
 *
 * WARN: non-default cudf::aggregation arguments are ignored. The default constructor is used
 *       always. This also means that we only support aggregations that have a default constructor!
 *
 * @param table The table to group and aggregate.
 * @param keys The names of the columns whose rows act as the groupby keys.
 * @param column_aggregations A vector of column aggregations to perform. Each column aggregation
 * produces a column in the output table by performing an aggregation-kind on a column in `table`.
 * It consist of a tuple: `(<input-column-name>, <aggregation-kind>, <output-column-name>)`. E.g.
 * `("x", SUM, "sum-of-x")}` will produce a column named "sum-of-x" in the output table, which, for
 * each groupby key, has a row that contains the sum of the values in the column "x". Multiple
 * column aggregations can share the same input column but all output columns must be unique and not
 * conflict with the name of the key columns.
 * @return A new logical table that contains the key columns and the aggregated columns using the
 * output column names and order specified in `column_aggregations`.
 */
LogicalTable groupby_aggregation(
  const LogicalTable& table,
  const std::vector<std::string>& keys,
  const std::vector<std::tuple<std::string, std::string, std::string>>& column_aggregations);

}  // namespace legate::dataframe
