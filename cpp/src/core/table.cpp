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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cudf/column/column_view.hpp>

#include <legate_dataframe/core/table.hpp>
#include <type_traits>

namespace legate::dataframe {

LogicalTable::LogicalTable(std::vector<LogicalColumn> columns,
                           std::map<std::string, size_t> column_names)
  : columns_{std::move(columns)}, column_names_{std::move(column_names)}
{
  if (column_names_.size() != columns_.size()) {
    throw std::invalid_argument("The size of `column_names` and `columns` must be equal");
  }
  if (columns_.size() == 0 || unbound()) { return; }
  auto n = columns_.cbegin()->num_rows();
  if (std::any_of(columns_.cbegin(), columns_.cend(), [n](const LogicalColumn& col) {
        return col.num_rows() != n;
      })) {
    throw std::invalid_argument("all columns in a table must have the same number of rows");
  }
}

namespace {
std::map<std::string, size_t> column_names_vector2map(const std::vector<std::string>& column_names)
{
  // enumerate the column names
  std::map<std::string, size_t> ret;
  for (const auto& name : column_names) {
    const auto [_, success] = ret.insert({name, ret.size()});
    if (!success) { throw std::invalid_argument("all column names must be unique"); }
  }
  return ret;
}
}  // namespace

LogicalTable::LogicalTable(std::vector<LogicalColumn> columns,
                           const std::vector<std::string>& column_names)
  : LogicalTable(std::move(columns), column_names_vector2map(column_names))
{
}

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

size_t LogicalTable::num_rows() const
{
  if (unbound()) {
    throw std::runtime_error("the num_rows of an unbound LogicalTable is undefined");
  }
  if (num_columns() == 0) { return 0; }
  return columns_.cbegin()->num_rows();
}

bool LogicalTable::unbound() const
{
  // If one of the underlying columns are unbound, the table as a whole is unbound.
  return std::any_of(
    columns_.cbegin(), columns_.cend(), [](const LogicalColumn& col) { return col.unbound(); });
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

std::string LogicalTable::repr(size_t max_num_items_ptr_column) const
{
  std::stringstream ss;
  const auto names = get_column_name_vector();
  ss << "LogicalTable(columns={";
  for (size_t i = 0; i < columns_.size(); ++i) {
    ss << names.at(i) + ": " + columns_.at(i).repr(max_num_items_ptr_column) << ", ";
  }
  ss << "\b\b})";  // use two ANSI backspace characters '\b' to overwrite the final ','
  return ss.str();
}

namespace argument {
std::vector<legate::Variable> add_next_input(legate::AutoTask& task,
                                             const LogicalTable& tbl,
                                             bool broadcast)
{
  // Send columns manually to avoid thousands of scalars for the dtypes.
  std::vector<legate::Variable> ret;
  std::vector<std::underlying_type_t<cudf::type_id>> type_ids;
  for (int i = 0; i < tbl.num_columns(); ++i) {
    auto col = tbl.get_column(i);
    type_ids.push_back(static_cast<std::underlying_type_t<cudf::type_id>>(col.cudf_type().id()));
    // Currently, tables are never scalar, so no need to add broadcast in that case.
    auto var = task.add_input(col.get_logical_array());
    if (broadcast) { task.add_constraint(legate::broadcast(var, {0})); }
    ret.push_back(var);
  }
  // Now add the column dtypes (this also gives us the number of columns).
  // Scalars are added separately, so order shouldn't matter.
  add_next_scalar_vector(task, type_ids);
  add_alignment_constraints(task, ret);
  return ret;
}

std::vector<legate::Variable> add_next_output(legate::AutoTask& task, const LogicalTable& tbl)
{
  // Send columns manually to avoid thousands of scalars for the dtypes.
  std::vector<legate::Variable> ret;
  std::vector<std::underlying_type_t<cudf::type_id>> type_ids;
  for (int i = 0; i < tbl.num_columns(); ++i) {
    auto col = tbl.get_column(i);
    type_ids.push_back(static_cast<std::underlying_type_t<cudf::type_id>>(col.cudf_type().id()));
    // Currently, tables are never scalar, so no need to add broadcast in that case.
    ret.push_back(task.add_output(col.get_logical_array()));
  }
  // Now add the column dtypes (this also gives us the number of columns).
  // Scalars are added separately, so order shouldn't matter.
  add_next_scalar_vector(task, type_ids);
  add_alignment_constraints(task, ret);
  return ret;
}

}  // namespace argument

}  // namespace legate::dataframe
