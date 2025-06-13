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
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/unary.hpp>

#include <legate.h>
#include <legate/cuda/cuda.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/core/transposed_copy.cuh>

#include <legate_dataframe/parquet.hpp>

namespace legate::dataframe::task {

class ParquetWrite : public Task<ParquetWrite, OpCode::ParquetWrite> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetWrite}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

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
    cudf::io::write_parquet(options, context.get_task_stream());
  }
};

class ParquetRead : public Task<ParquetRead, OpCode::ParquetRead> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetRead}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

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
      tbl_arg.move_into(cudf::concatenate(table_views, context.get_task_stream()));
    }
  }
};

class ParquetReadArray : public Task<ParquetReadArray, OpCode::ParquetReadArray> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetReadArray}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto file_paths = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns    = argument::get_next_scalar_vector<std::string>(ctx);
    const auto nrows      = argument::get_next_scalar_vector<size_t>(ctx);
    auto null_value       = ctx.get_next_scalar_arg();
    auto out              = ctx.get_next_output_arg();

    auto expected_type_id = to_cudf_type_id(out.type().code());

    if (file_paths.size() != nrows.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }
    const size_t ncols = columns.size();
    if (columns.size() != out.shape<2>().hi[1] + 1) {
      throw std::runtime_error("internal error: columns size and result shape mismatch");
    }

    // Iterate through the file and nrow list and read as many rows from the
    // files as this rank should read while skipping those of the other tasks.
    std::vector<std::unique_ptr<cudf::table>> tables;
    size_t total_rows_seen = 0;  // offset into the current file
    for (size_t i = 0; i < file_paths.size(); i++) {
      auto file_rows = nrows[i];

      if (total_rows_seen + file_rows < out.shape<2>().lo[0]) {
        // All of this files rows belong to earlier ranks.
        total_rows_seen += file_rows;
        continue;
      } else if (total_rows_seen > out.shape<2>().hi[0]) {
        // We have read all rows for this chunk.
        break;
      }

      // start and end (exclusive) rows within this particular file:
      auto start =
        out.shape<2>().lo[0] > total_rows_seen ? out.shape<2>().lo[0] - total_rows_seen : 0;
      auto end = std::min<size_t>(out.shape<2>().hi[0] - total_rows_seen + 1, file_rows);

      while (start < end) {
        // Read a few hundred MiB at a time (assumes datatype isn't very narrow)
        auto nrows_to_read = std::min<size_t>(end - start, ((1 << 25) + ncols - 1) / ncols);

        auto src = cudf::io::source_info(file_paths[i]);
        auto opt = cudf::io::parquet_reader_options::builder(src);
        opt.columns(columns);
        opt.skip_rows(start);
        opt.num_rows(nrows_to_read);

        auto tbl = cudf::io::read_parquet(opt, ctx.stream(), ctx.mr()).tbl;
        /* Check if all columns are of the right type and cast them if not. */
        auto column_vec = tbl->release();
        for (auto& col : column_vec) {
          if (col->type().id() != expected_type_id) {
            col =
              cudf::cast(col->view(), cudf::data_type{expected_type_id}, ctx.stream(), ctx.mr());
          }
        }
        auto cast_tbl = cudf::table(std::move(column_vec));

        // Write to output array, this is a transposed copy.
        copy_into_tranposed(ctx, out, cast_tbl.view(), start + total_rows_seen, null_value);

        start += nrows_to_read;
      }
      total_rows_seen += file_rows;
    }
  }
};

}  // namespace legate::dataframe::task

namespace legate::dataframe {

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::ParquetWrite::register_variants();
  legate::dataframe::task::ParquetRead::register_variants();
  legate::dataframe::task::ParquetReadArray::register_variants();
  return 0;
}();

struct ParquetReadInfo {
  std::vector<std::string> file_paths;
  std::vector<std::string> column_names;
  std::vector<size_t> nrows;
  size_t nrows_total;
  std::unique_ptr<cudf::table> prototype_table;
};

ParquetReadInfo get_parquet_info(const std::string& glob_string,
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

  // Validate column names if specified
  if (columns.has_value()) {
    std::set<std::string> names_in_table{column_names.begin(), column_names.end()};
    for (auto col : columns.value()) {
      if (names_in_table.count(col) == 0) {
        throw std::invalid_argument("column was not found in parquet file: " + std::string(col));
      }
    }
  }

  return {std::move(file_paths),
          std::move(column_names),
          std::move(nrows),
          nrows_total,
          std::move(result.tbl)};
}

}  // namespace

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
  auto info        = get_parquet_info(glob_string, columns);
  LogicalTable ret = LogicalTable::empty_like(*info.prototype_table, info.column_names);

  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetRead::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, info.file_paths);
  argument::add_next_scalar_vector(task, info.column_names);
  argument::add_next_scalar_vector(task, info.nrows);
  argument::add_next_scalar(task, info.nrows_total);
  argument::add_next_output(task, ret);
  argument::add_parallel_launch_task(task);
  runtime->submit(std::move(task));
  return ret;
}

legate::LogicalArray parquet_read_array(const std::string& glob_string,
                                        const std::optional<std::vector<std::string>>& columns,
                                        const legate::Scalar& null_value,
                                        const std::optional<legate::Type>& type)
{
  auto runtime = legate::Runtime::get_runtime();
  auto info    = get_parquet_info(glob_string, columns);

  legate::Type legate_type = legate::null_type();

  if (!type.has_value()) {
    auto cudf_type = info.prototype_table->get_column(0).type();
    if (!cudf::is_numeric(cudf_type)) {
      throw std::invalid_argument("only numeric columns are supported for parquet_read_array");
    }
    for (auto&& col : info.prototype_table->view()) {
      if (col.type().id() != cudf_type.id()) {
        throw std::invalid_argument("all columns must have the same type");
      }
    }

    legate_type = to_legate_type(cudf_type.id());
  } else {
    legate_type = type.value();

    auto cudf_type = cudf::data_type{to_cudf_type_id(legate_type.code())};
    for (auto&& col : info.prototype_table->view()) {
      if (!cudf::is_supported_cast(col.type(), cudf_type)) {
        throw std::invalid_argument("Cannot cast all columns to specified type");
      }
    }
  }

  auto nullable = null_value.type().code() == Type::Code::NIL;
  if (!nullable) {
    if (null_value.type() != legate_type) {
      throw std::invalid_argument("null value must be null or have the same type as the result");
    }
  }

  auto ret =
    runtime->create_array({info.nrows_total, info.column_names.size()}, legate_type, nullable);

  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetReadArray::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, info.file_paths);
  argument::add_next_scalar_vector(task, info.column_names);
  argument::add_next_scalar_vector(task, info.nrows);
  argument::add_next_scalar(task, null_value);

  auto var = task.add_output(ret);
  task.add_constraint(legate::broadcast(var, {1}));
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
