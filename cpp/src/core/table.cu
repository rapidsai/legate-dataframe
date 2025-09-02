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

#include <vector>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

namespace {
std::vector<LogicalColumn> from_cudf_table(const cudf::table_view& cudf_table,
                                           rmm::cuda_stream_view stream)
{
  std::vector<LogicalColumn> ret;
  for (const cudf::column_view& col : cudf_table) {
    ret.emplace_back(col, stream);
  }
  return ret;
}
}  // namespace

LogicalTable::LogicalTable(cudf::table_view cudf_table,
                           const std::vector<std::string>& column_names,
                           rmm::cuda_stream_view stream)
  : LogicalTable(from_cudf_table(cudf_table, stream), column_names)
{
}

std::unique_ptr<cudf::table> LogicalTable::get_cudf(rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr) const
{
  if (unbound()) {
    throw std::runtime_error("cannot get a cudf table from an unbound LogicalTable");
  }
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(columns_.size());
  for (const auto& col : columns_) {
    cols.push_back(col.get_cudf(stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace legate::dataframe
