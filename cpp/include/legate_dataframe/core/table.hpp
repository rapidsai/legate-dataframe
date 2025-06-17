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

#include <map>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/task_argument.hpp>

namespace legate::dataframe {

/**
 * @brief Collection of logical columns
 *
 * The order of the collection of columns is preserved. Use `.get_column` and `.get_columns`
 * to access individual columns.
 *
 * Unlike libcudf, the columns in a `LogicalTable` have names, which makes it possible to retrieve
 * columns by name using `.get_column()`. Additionally, when reading and writing tables to/from
 * files, the column names are read and written automatically.
 * Since libcudf doesn't use column names, the `PhysicalTable` doesn't either. It is a strict
 * `LogicalTable` feature.
 *
 * Notice, the table doesn't _own_ the columns, a column can be in multiple tables.
 *
 */
class LogicalTable {
 public:
  /**
   * @brief Create an undefined table only to used for copying and moving
   *
   * This ctor is only here because of Cython
   */
  LogicalTable() = default;

  /**
   * @brief Create a table from a vector of columns
   *
   * @param column_names A bijective mapping of column names to indices in `columns`. That is,
   * each column index is mapped to from exactly one column name.
   */
  LogicalTable(std::vector<LogicalColumn> columns, std::map<std::string, size_t> column_names);

  /**
   * @brief Create a table from a vector of columns.
   *
   * @throw invalid_argument if the column names are not unique.
   * @param column_names A vector of column names given in the same order as `columns`.
   */
  LogicalTable(std::vector<LogicalColumn> columns, const std::vector<std::string>& column_names);

  /**
   * @brief Create a table from a local cudf table
   *
   * This call blocks the client's control flow and scatter the data to all
   * legate nodes.
   *
   * @param cudf_table The local cuDF table to copy into a new logical table
   * @param column_names A vector of column names given in the same order as in `cudf_table`.
   * @param stream CUDA stream used for device memory operations
   */
  LogicalTable(cudf::table_view cudf_table,
               const std::vector<std::string>& column_names,
               rmm::cuda_stream_view stream = cudf::get_default_stream());

  /**
   * @brief Create a new unbounded table from an existing table
   *
   * @param other The prototype table
   * @return The new unbounded table with empty columns like `other`
   */
  static LogicalTable empty_like(const LogicalTable& other)
  {
    std::vector<LogicalColumn> columns;
    columns.reserve(other.columns_.size());
    for (const auto& col : other.columns_) {
      columns.emplace_back(LogicalColumn::empty_like(col));
    }
    return LogicalTable(std::move(columns), other.get_column_names());
  }

  /**
   * @brief Create a new unbounded table from an existing cudf table
   *
   * @param other The prototype table
   * @param column_names A vector of column names given in the same order as in `other`.
   * @return The new unbounded table with empty columns like `other`
   */
  static LogicalTable empty_like(const cudf::table_view& other,
                                 const std::vector<std::string>& column_names)
  {
    std::vector<LogicalColumn> columns;
    columns.reserve(other.num_columns());
    for (int i = 0; i < other.num_columns(); ++i) {
      columns.emplace_back(LogicalColumn::empty_like(other.column(i)));
    }
    return LogicalTable(std::move(columns), column_names);
  }

 public:
  LogicalTable(const LogicalTable& other)            = default;
  LogicalTable& operator=(const LogicalTable& other) = default;
  LogicalTable(LogicalTable&& other)                 = default;
  LogicalTable& operator=(LogicalTable&& other)      = default;

 public:
  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] int32_t num_columns() const { return columns_.size(); }

  /**
   * @brief Returns the number of rows
   *
   * @throw std::runtime_error if table is unbound.
   * @return The number of rows
   */
  [[nodiscard]] size_t num_rows() const;

  /**
   * @brief Returns a reference to the specified column
   *
   * @throws std::out_of_range If i is out of the range [0, num_columns)
   *
   * @param column_index Index of the desired column
   * @return A reference to the desired column
   */
  [[nodiscard]] LogicalColumn& get_column(size_t column_index) { return columns_.at(column_index); }

  /**
   * @brief Returns a const reference to the specified column
   *
   * @throws std::out_of_range If i is out of the range [0, num_columns)
   *
   * @param column_index Index of the desired column
   * @return A const reference to the desired column
   */
  [[nodiscard]] const LogicalColumn& get_column(size_t column_index) const
  {
    return columns_.at(column_index);
  }

  /**
   * @brief Returns a reference to the specified column
   *
   * @throws std::out_of_range If `column_name` doesn't exist
   *
   * @param column_index Name of the desired column
   * @return A reference to the desired column
   */
  [[nodiscard]] LogicalColumn& get_column(const std::string& column_name)
  {
    return get_column(column_names_.at(column_name));
  }

  /**
   * @brief Returns a const reference to the specified column
   *
   * @throws std::out_of_range If `column_name` doesn't exist
   *
   * @param column_index Name of the desired column
   * @return A const reference to the desired column
   */
  [[nodiscard]] const LogicalColumn& get_column(const std::string& column_name) const
  {
    return get_column(column_names_.at(column_name));
  }

  /**
   * @brief Returns a const reference to a vector of columns
   *
   * @return A const reference to the columns of this table
   */
  [[nodiscard]] const std::vector<LogicalColumn>& get_columns() const { return columns_; }

  /**
   * @brief Returns a const reference to the mapping of column names to indices
   *
   * @return A const reference to the name -> index mapping
   */
  [[nodiscard]] const std::map<std::string, size_t>& get_column_names() const noexcept
  {
    return column_names_;
  }

  /**
   * @brief Returns a copy of the column names order by column indices
   *
   * @return A copy of the column names
   */
  [[nodiscard]] std::vector<std::string> get_column_name_vector() const
  {
    std::vector<std::string> ret(get_column_names().size());
    for (const auto& [name, idx] : get_column_names()) {
      ret.at(idx) = name;
    }
    return ret;
  }

  /**
   * @brief Returns a new table with the set of specified columns.
   *
   * @throw std::out_of_range if any element in `column_indices` is outside [0, num_columns())
   *
   * @param column_indices Indices of columns in the table (the order matters).
   * @return A logical table consisting of columns from the original table
   * specified by the elements of `column_indices`
   */
  [[nodiscard]] LogicalTable select(const std::vector<size_t>& column_indices) const
  {
    std::vector<LogicalColumn> columns;
    std::map<std::string, size_t> names;
    const auto col_idx_to_name = get_column_name_vector();
    for (size_t idx : column_indices) {
      columns.push_back(get_column(idx));
      names.insert({col_idx_to_name.at(idx), names.size()});
    }
    return LogicalTable(std::move(columns), std::move(names));
  };

  /**
   * @brief Returns a new table with the set of specified columns.
   *
   * @throw std::out_of_range if any name in `column_names` isn't found.
   *
   * @param column_names Column names in the table (the order matters).
   * @return A logical table consisting of columns from the original table
   * specified by the names in `column_names`
   */
  [[nodiscard]] LogicalTable select(const std::vector<std::string>& column_names) const
  {
    std::vector<LogicalColumn> columns;
    std::map<std::string, size_t> names;
    for (const auto& name : column_names) {
      columns.push_back(get_column(name));
      names.insert({name, names.size()});
    }
    return LogicalTable(std::move(columns), std::move(names));
  };

  /**
   * @brief Offload all columns to the specified target memory.
   *
   * This method offloads the underlying data to the specified target memory.
   * The purpose of this is to free up GPU memory resources.
   * See `legate::LogicalArray::offload_to` for more information.
   *
   * @param target_mem The `legate::mapping::StoreTarget` target memory.
   * This will be `legate::mapping::StoreTarget::SYSMEM` to move data to the CPU.
   */
  void offload_to(legate::mapping::StoreTarget target_mem) const
  {
    for (const auto& col : columns_) {
      return col.offload_to(target_mem);
    }
  }

  /**
   * @brief Indicates whether the table is unbound
   *
   * A table is consider unbound if one of its columns is unbound.
   *
   * @return true The table is unbound
   * @return false The table is bound
   */
  [[nodiscard]] bool unbound() const;

  /**
   * @brief Copy the logical table into a local cudf table

   * This call blocks the client's control flow and fetches the data for the
   * whole table to the current node.
   *
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations.
   * @return cudf table, which own the data
   */
  std::unique_ptr<cudf::table> get_cudf(
    rmm::cuda_stream_view stream        = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Copy the logical table into a local arrow table

   * This call blocks the client's control flow and fetches the data for the
   * whole table to the current node.
   *
   * @return arrow table, which own the data
   */
  std::shared_ptr<arrow::Table> get_arrow() const;

  /**
   * @brief Return a printable representational string
   *
   * @param max_num_items Maximum number of items to include before items are abbreviated.
   * @return Printable representational string
   */
  std::string repr(size_t max_num_items_ptr_column = 30) const;

 private:
  std::vector<LogicalColumn> columns_;
  std::map<std::string, size_t> column_names_;
};

namespace task {

/**
 * @brief Local physical table used in tasks
 */
class PhysicalTable {
 public:
  PhysicalTable(std::vector<PhysicalColumn> columns) : columns_{std::move(columns)} {}

 public:
  PhysicalTable(const PhysicalTable& other)            = default;
  PhysicalTable& operator=(const PhysicalTable& other) = default;
  PhysicalTable(PhysicalTable&& other)                 = default;
  PhysicalTable& operator=(PhysicalTable&& other)      = default;

 public:
  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] int32_t num_columns() const { return columns_.size(); }

  /**
   * @brief Return a cudf table view of this physical table
   *
   * NB: The physical table MUST outlive the returned view thus it is UB to do some-
   *     thing like `argument::get_next_input<PhysicalTable>(ctx).table_view();`
   *
   * @throw std::runtime_error if table is unbound.
   * @return A new table view
   */
  cudf::table_view table_view() const
  {
    std::vector<cudf::column_view> cols;
    cols.reserve(columns_.size());
    for (const auto& col : columns_) {
      cols.push_back(col.column_view());
    }
    return cudf::table_view(std::move(cols));
  }

  /**
   * @brief Creates an Arrow Table view using the specified column names.
   *
   * @param column_names
   * @return std::shared_ptr<arrow::Table> A shared pointer to the newly created Arrow Table.
   *
   * @throws std::runtime_error If the number of provided column names does not match the number of
   * columns.
   */
  std::shared_ptr<arrow::Table> arrow_table_view(const std::vector<std::string>& column_names) const
  {
    if (static_cast<std::size_t>(column_names.size()) != columns_.size()) {
      throw std::runtime_error("LogicalTable.arrow_table_view(): number of columns mismatch " +
                               std::to_string(columns_.size()) +
                               " != " + std::to_string(column_names.size()));
    }
    std::vector<std::shared_ptr<arrow::Array>> cols;
    cols.reserve(columns_.size());
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (std::size_t i = 0; i < columns_.size(); i++) {
      const auto& col = columns_[i];
      cols.push_back(col.arrow_array_view());
      fields.push_back(arrow::field(column_names[i], col.arrow_type()));
    }
    return arrow::Table::Make(arrow::schema(fields), std::move(cols));
  }

  /**
   * @brief Move local cudf columns into this unbound physical table
   *
   * @param columns The cudf columns to move
   */
  void move_into(std::vector<std::unique_ptr<cudf::column>> columns)
  {
    if (columns.size() != columns_.size()) {
      throw std::runtime_error("LogicalTable.move_into(): number of columns mismatch " +
                               std::to_string(columns_.size()) +
                               " != " + std::to_string(columns.size()));
    }
    for (size_t i = 0; i < columns.size(); ++i) {
      columns_[i].move_into(std::move(columns[i]));
    }
  }

  /**
   * @brief Move local arrow arrays into this unbound physical table
   *
   * @param columns The arrow arrays to move
   */
  void move_into(std::shared_ptr<arrow::Table> table)
  {
    if (static_cast<std::size_t>(table->num_columns()) != columns_.size()) {
      throw std::runtime_error("LogicalTable.move_into(): number of columns mismatch " +
                               std::to_string(columns_.size()) +
                               " != " + std::to_string(table->num_columns()));
    }
    // Component chunked arrays must be converted to contiguous arrays
    auto combined = table->CombineChunks().ValueOrDie();
    for (int i = 0; i < combined->num_columns(); ++i) {
      auto chunked_array = combined->column(i);
      if (chunked_array->num_chunks() != 1) {
        throw std::runtime_error("LogicalTable.move_into(): expected an array with 1 chunk.");
      }
      columns_[i].move_into(chunked_array->chunk(0));
    }
  }

  /**
   * @brief Move local cudf table into this unbound physical table
   *
   * @param table The cudf table to move
   */
  void move_into(std::unique_ptr<cudf::table> table) { move_into(table->release()); }

  /**
   * @brief Makes the unbound table empty. Valid only when the table is unbound.
   */
  void bind_empty_data() const
  {
    for (const auto& col : columns_) {
      col.bind_empty_data();
    }
  }

  /**
   * @brief Returns true if the data is partitioned.
   *
   * You can use this to check whether a column is partitioned, please see
   * `legate::PhysicalStore::is_partitioned` for more information.
   * This can be used to check whether a column is broadcasted (i.e. partitioned
   * is false), meaning that all workers see the same data.
   *
   * This function relies on tables always adding an alignment constraint.
   *
   * @throw std::out_of_range if the table doesn't have at least one column.
   * @return true if data is partitioned.
   */
  [[nodiscard]] bool is_partitioned() const { return columns_.at(0).is_partitioned(); }

  /**
   * @brief Releases ownership of the `column`s by returning a vector of
   * the constituent columns.
   *
   * After `release()`, `num_columns() == 0`.
   *
   * @returns A vector of the constituent columns
   */
  std::vector<PhysicalColumn> release() { return std::move(columns_); }

  /**
   * @brief Returns the column dtypes
   *
   * @return The column dtypes.
   */
  [[nodiscard]] std::vector<cudf::data_type> cudf_types() const
  {
    std::vector<cudf::data_type> dtypes;

    for (const auto& col : columns_) {
      dtypes.push_back(col.cudf_type());
    }
    return dtypes;
  }

  [[nodiscard]] std::vector<std::shared_ptr<arrow::DataType>> arrow_types() const
  {
    std::vector<std::shared_ptr<arrow::DataType>> dtypes;
    for (const auto& col : columns_) {
      dtypes.push_back(col.arrow_type());
    }
    return dtypes;
  }

 private:
  std::vector<PhysicalColumn> columns_;
};
}  // namespace task

namespace argument {

/**
 * @brief Add a logical table to the next input task argument
 *
 * This adds alignment constraints to all logical columns within the table.
 * This should match a call to `get_next_input<PhysicalTable>()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @param task The legate task to add the argument.
 * @param tbl The logical table to add as the next task argument.
 * @param broadcast If set to true, each worker is guaranteed to get a copy
 * of the data.
 */
std::vector<legate::Variable> add_next_input(legate::AutoTask& task,
                                             const LogicalTable& tbl,
                                             bool broadcast = false);

/**
 * @brief Add a logical table to the next output task argument
 *
 * This adds alignment constraints to all logical columns within the table.
 * This should match a call to `get_next_input<PhysicalTable>()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @param task The legate task to add the argument.
 * @param tbl The logical table to add as the next task argument.
 */
std::vector<legate::Variable> add_next_output(legate::AutoTask& task, const LogicalTable& tbl);

template <>
inline task::PhysicalTable get_next_input<task::PhysicalTable>(TaskContext& ctx)
{
  auto num_columns = get_next_scalar<int32_t>(ctx);
  std::vector<task::PhysicalColumn> cols;
  cols.reserve(num_columns);
  for (auto i = 0; i < num_columns; ++i) {
    cols.push_back(argument::get_next_input<task::PhysicalColumn>(ctx));
  }
  return task::PhysicalTable(std::move(cols));
}

template <>
inline task::PhysicalTable get_next_output<task::PhysicalTable>(TaskContext& ctx)
{
  auto num_columns = get_next_scalar<int32_t>(ctx);
  std::vector<task::PhysicalColumn> cols;
  cols.reserve(num_columns);
  for (auto i = 0; i < num_columns; ++i) {
    cols.push_back(argument::get_next_output<task::PhysicalColumn>(ctx));
  }
  return task::PhysicalTable(std::move(cols));
}

}  // namespace argument

}  // namespace legate::dataframe
