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

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cudf/column/column_view.hpp>

#include <legate_dataframe/core/table.hpp>

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

std::shared_ptr<arrow::Table> LogicalTable::get_arrow() const
{
  if (unbound()) {
    throw std::runtime_error("cannot get an arrow table from an unbound LogicalTable");
  }
  std::vector<std::shared_ptr<arrow::Array>> cols;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  auto names = this->get_column_name_vector();
  for (std::size_t i = 0; i < columns_.size(); i++) {
    cols.push_back(columns_.at(i).get_arrow());
    fields.push_back(arrow::field(names.at(i), columns_.at(i).arrow_type()));
  }
  return arrow::Table::Make(arrow::schema(fields), std::move(cols));
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
  std::vector<legate::Variable> ret;
  // First we add number of columns
  add_next_scalar(task, tbl.num_columns());
  // Then we add each column
  for (const auto& col : tbl.get_columns()) {
    ret.push_back(add_next_input(task, col, broadcast));
  }
  add_alignment_constraints(task, ret);
  return ret;
}

std::vector<legate::Variable> add_next_output(legate::AutoTask& task, const LogicalTable& tbl)
{
  std::vector<legate::Variable> ret;
  // First we add number of columns
  add_next_scalar(task, tbl.num_columns());
  // Then we add each column
  for (int i = 0; i < tbl.num_columns(); ++i) {
    ret.push_back(add_next_output(task, tbl.get_column(i)));
  }
  add_alignment_constraints(task, ret);
  return ret;
}

}  // namespace argument

}  // namespace legate::dataframe
