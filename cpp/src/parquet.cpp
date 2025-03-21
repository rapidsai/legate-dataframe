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

#include <filesystem>
#include <stdexcept>
#include <vector>

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>

#include <legate.h>
#include <legate/cuda/cuda.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_context.hpp>

#include <legate_dataframe/parquet.hpp>

namespace legate::dataframe::task {

class ParquetWrite : public Task<ParquetWrite, OpCode::ParquetWrite> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetWrite}};

  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const std::string dirpath  = argument::get_next_scalar<std::string>(ctx);
    const auto column_names    = argument::get_next_scalar_vector<std::string>(ctx);
    const auto table           = argument::get_next_input<PhysicalTable>(ctx);
    const std::string filepath = dirpath + "/part." + std::to_string(ctx.rank) + ".parquet";
    const auto tbl             = table.table_view();

    auto dest    = cudf::io::sink_info(filepath);
    auto options = cudf::io::parquet_writer_options::builder(dest, tbl);
    cudf::io::table_input_metadata metadata(tbl);

    // Set column names
    for (size_t i = 0; i < metadata.column_metadata.size(); i++) {
      metadata.column_metadata.at(i).set_name(column_names.at(i));
    }
    options.metadata(metadata);
    cudf::io::write_parquet(options, ctx.stream());
  }
};

class ParquetRead : public Task<ParquetRead, OpCode::ParquetRead> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetRead}};

  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const auto file_paths  = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns     = argument::get_next_scalar_vector<std::string>(ctx);
    const auto nrows       = argument::get_next_scalar_vector<size_t>(ctx);
    const auto nrows_total = argument::get_next_scalar<size_t>(ctx);
    PhysicalTable tbl_arg  = argument::get_next_output<PhysicalTable>(ctx);
    argument::get_parallel_launch_task(ctx);

    if (file_paths.size() != nrows.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }

    auto [my_rows_offset, my_num_rows] = evenly_partition_work(nrows_total, ctx.rank, ctx.nranks);

    // Iterate through the file and nrow list and read as many rows from the
    // files as this rank should read while skipping those of the other tasks.
    std::vector<std::unique_ptr<cudf::table>> tables;
    size_t total_rows_seen = 0;
    for (size_t i = 0; i < file_paths.size() && my_num_rows > 0; i++) {
      auto file_rows = nrows[i];

      if (total_rows_seen + file_rows < my_rows_offset) {
        // All of this files rows belong to earlier ranks.
        total_rows_seen += file_rows;
        continue;
      }
      // Calculate offset and rows to read from this file.
      auto file_rows_offset  = my_rows_offset - total_rows_seen;
      auto file_rows_to_read = std::min(file_rows - file_rows_offset, my_num_rows);

      auto src = cudf::io::source_info(file_paths[i]);
      auto opt = cudf::io::parquet_reader_options::builder(src);
      opt.columns(columns);
      opt.skip_rows(file_rows_offset);
      opt.num_rows(file_rows_to_read);
      tables.emplace_back(std::move(cudf::io::read_parquet(opt, ctx.stream(), ctx.mr()).tbl));

      my_num_rows -= file_rows_to_read;
      my_rows_offset += file_rows_to_read;
      total_rows_seen += file_rows;
    }

    // Concatenate tables and move the result to the output table
    if (tables.size() == 0) {
      tbl_arg.bind_empty_data();
    } else if (tables.size() == 1) {
      tbl_arg.move_into(std::move(tables.back()));
    } else {
      std::vector<cudf::table_view> table_views;
      for (const auto& table : tables) {
        table_views.push_back(table->view());
      }
      tbl_arg.move_into(cudf::concatenate(table_views, ctx.stream(), ctx.mr()));
    }
  }
};

}  // namespace legate::dataframe::task

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::ParquetWrite::register_variants();
  legate::dataframe::task::ParquetRead::register_variants();
  return 0;
}();

}  // namespace

namespace legate::dataframe {

void parquet_write(LogicalTable& tbl, const std::string& dirpath)
{
  std::filesystem::create_directories(dirpath);
  if (!std::filesystem::is_empty(dirpath)) {
    throw std::invalid_argument("if path exist, it must be an empty directory");
  }
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetWrite::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, dirpath);
  argument::add_next_scalar_vector(task, tbl.get_column_name_vector());
  argument::add_next_input(task, tbl);
  runtime->submit(std::move(task));
}

LogicalTable parquet_read(const std::string& glob_string,
                          const std::optional<std::vector<std::string>>& columns)
{
  std::vector<std::string> file_paths = parse_glob(glob_string);
  if (file_paths.empty()) { throw std::invalid_argument("no parquet files specified"); }
  // We read the meta data from the first file
  auto source  = cudf::io::source_info(file_paths[0]);
  auto options = cudf::io::parquet_reader_options::builder(source).num_rows(1);
  if (columns.has_value()) { options.columns(columns.value()); }
  auto result = cudf::io::read_parquet(options);

  // Get the column names
  std::vector<std::string> column_names;
  column_names.reserve(result.metadata.schema_info.size());
  for (const auto& column_name_info : result.metadata.schema_info) {
    column_names.push_back(column_name_info.name);
  }

  // Get the number of rows in each file:
  std::vector<size_t> nrows;
  size_t nrows_total = 0;
  nrows.reserve(file_paths.size());
  for (const auto& path : file_paths) {
    auto source   = cudf::io::source_info(path);
    auto metadata = cudf::io::read_parquet_metadata(source);
    nrows.push_back(metadata.num_rows());
    nrows_total += metadata.num_rows();
  }

  LogicalTable ret = LogicalTable::empty_like(*result.tbl, column_names);

  // cudf doesn't raise if names are missing, so check now (just to have a map)
  auto names_in_table = ret.get_column_names();
  if (columns.has_value() && columns.value().size() != names_in_table.size()) {
    for (auto col : columns.value()) {
      if (names_in_table.count(col) == 0) {
        throw std::invalid_argument("column was not found in parquet file: " + std::string(col));
      }
    }
    // Should never reach here:
    throw std::invalid_argument("not all columns found in parquet file.");
  }

  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetRead::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, file_paths);
  argument::add_next_scalar_vector(task, ret.get_column_name_vector());
  argument::add_next_scalar_vector(task, nrows);
  argument::add_next_scalar(task, nrows_total);
  argument::add_next_output(task, ret);
  argument::add_parallel_launch_task(task);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
