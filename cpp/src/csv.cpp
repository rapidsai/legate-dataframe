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

#include <filesystem>
#include <stdexcept>
#include <vector>

#include <cudf/concatenate.hpp>
#include <cudf/io/csv.hpp>
#include <legate.h>
#include <rmm/device_buffer.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_context.hpp>

#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <legate_dataframe/csv.hpp>

namespace legate::dataframe::task {

class CSVWrite : public Task<CSVWrite, OpCode::CSVWrite> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::CSVWrite}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const std::string dirpath  = argument::get_next_scalar<std::string>(ctx);
    const auto column_names    = argument::get_next_scalar_vector<std::string>(ctx);
    const auto tbl             = argument::get_next_input<PhysicalTable>(ctx);
    const std::string filepath = dirpath + "/part." + std::to_string(ctx.rank) + ".csv";
    const auto delimiter       = static_cast<char>(argument::get_next_scalar<int32_t>(ctx));

    auto arrow_table        = tbl.arrow_table_view(column_names);
    auto outfile            = arrow::io::FileOutputStream::Open(filepath).ValueOrDie();
    auto write_options      = arrow::csv::WriteOptions::Defaults();
    write_options.delimiter = delimiter;

    auto csv_writer =
      arrow::csv::MakeCSVWriter(outfile, arrow_table->schema(), write_options).ValueOrDie();
    auto status = csv_writer->WriteTable(*arrow_table);
    status      = csv_writer->Close();
    if (!status.ok()) {
      throw std::runtime_error("Failed to write CSV file: " + status.ToString());
    }
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    TaskMemoryResource mr;
    const std::string dirpath  = argument::get_next_scalar<std::string>(ctx);
    const auto column_names    = argument::get_next_scalar_vector<std::string>(ctx);
    const auto tbl             = argument::get_next_input<PhysicalTable>(ctx);
    const std::string filepath = dirpath + "/part." + std::to_string(ctx.rank) + ".csv";
    const auto delimiter       = static_cast<char>(argument::get_next_scalar<int32_t>(ctx));

    auto dest    = cudf::io::sink_info(filepath);
    auto options = cudf::io::csv_writer_options::builder(dest, tbl.table_view(mr));
    options.names(column_names);
    options.inter_column_delimiter(delimiter);

    cudf::io::write_csv(options, context.get_task_stream());
  }
};

class CSVRead : public Task<CSVRead, OpCode::CSVRead> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{OpCode::CSVRead}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const auto launch_domain    = context.get_launch_domain();
    const auto file_paths       = argument::get_next_scalar_vector<std::string>(ctx);
    const auto all_column_names = argument::get_next_scalar_vector<std::string>(ctx);
    const auto use_cols_indexes = argument::get_next_scalar_vector<int>(ctx);
    const auto na_filter        = argument::get_next_scalar<bool>(ctx);
    const auto delimiter        = static_cast<char>(argument::get_next_scalar<int32_t>(ctx));
    const auto nbytes           = argument::get_next_scalar_vector<size_t>(ctx);
    const auto nbytes_total     = argument::get_next_scalar<size_t>(ctx);
    const auto read_header      = argument::get_next_scalar<bool>(ctx);
    PhysicalTable tbl_arg       = argument::get_next_output<PhysicalTable>(ctx);
    argument::get_parallel_launch_task(ctx);

    if (file_paths.size() != nbytes.size()) {
      throw std::runtime_error("internal error: file path and nbytes size mismatch");
    }

    auto dtypes = tbl_arg.arrow_types();

    std::vector<std::string> column_names;
    for (auto index : use_cols_indexes) {
      column_names.push_back(all_column_names[index]);
    }

    std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> dtypes_map;
    for (size_t i = 0; i < dtypes.size(); i++) {
      dtypes_map[column_names[i]] = dtypes[i];
    }
    // Assign one file to each rank for now
    auto [file_offset, num_files] = evenly_partition_work(file_paths.size(), ctx.rank, ctx.nranks);
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (size_t i = file_offset; i < file_offset + num_files; i++) {
      auto input = *arrow::io::ReadableFile::Open(file_paths[i]);

      auto read_options         = arrow::csv::ReadOptions::Defaults();
      read_options.use_threads  = false;
      read_options.column_names = all_column_names;
      read_options.skip_rows    = read_header ? 1 : 0;

      auto parse_options              = arrow::csv::ParseOptions::Defaults();
      parse_options.delimiter         = delimiter;
      auto convert_options            = arrow::csv::ConvertOptions::Defaults();
      convert_options.column_types    = dtypes_map;
      convert_options.include_columns = column_names;

      // Instantiate TableReader from input stream and options
      arrow::io::IOContext io_context                 = arrow::io::default_io_context();
      std::shared_ptr<arrow::csv::TableReader> reader = *arrow::csv::TableReader::Make(
        io_context, input, read_options, parse_options, convert_options);

      // Read table from CSV file
      tables.push_back(*reader->Read());
    }

    auto table = *arrow::ConcatenateTables(tables);

    // Concatenate tables
    tbl_arg.move_into(table);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    TaskMemoryResource mr;
    const auto file_paths       = argument::get_next_scalar_vector<std::string>(ctx);
    const auto all_column_names = argument::get_next_scalar_vector<std::string>(ctx);
    const auto use_cols_indexes = argument::get_next_scalar_vector<int>(ctx);
    const auto na_filter        = argument::get_next_scalar<bool>(ctx);
    const auto delimiter        = static_cast<char>(argument::get_next_scalar<int32_t>(ctx));
    const auto nbytes           = argument::get_next_scalar_vector<size_t>(ctx);
    const auto nbytes_total     = argument::get_next_scalar<size_t>(ctx);
    const auto read_header      = argument::get_next_scalar<bool>(ctx);
    PhysicalTable tbl_arg       = argument::get_next_output<PhysicalTable>(ctx);
    argument::get_parallel_launch_task(ctx);

    if (file_paths.size() != nbytes.size()) {
      throw std::runtime_error("internal error: file path and nbytes size mismatch");
    }

    auto [my_bytes_offset, my_num_bytes] =
      evenly_partition_work(nbytes_total, ctx.rank, ctx.nranks);

    auto dtypes = tbl_arg.cudf_types();

    std::vector<std::string> column_names;
    for (auto index : use_cols_indexes) {
      column_names.push_back(all_column_names[index]);
    }
    std::map<std::string, cudf::data_type> dtypes_map;
    for (size_t i = 0; i < dtypes.size(); i++) {
      dtypes_map[column_names[i]] = dtypes[i];
    }

    // Iterate through the file and nrow list and read as many rows from the
    // files as this rank should read while skipping those of the other tasks.
    std::vector<std::unique_ptr<cudf::table>> tables;
    size_t total_bytes_seen = 0;
    for (size_t i = 0; i < file_paths.size() && my_num_bytes > 0; i++) {
      auto file_bytes = nbytes[i];

      if (total_bytes_seen + file_bytes <= my_bytes_offset) {
        // All of this files bytes belong to earlier ranks.
        total_bytes_seen += file_bytes;
        continue;
      }
      // Calculate offset and bytes to read from this file.
      auto file_bytes_offset  = my_bytes_offset - total_bytes_seen;
      auto file_bytes_to_read = std::min(file_bytes - file_bytes_offset, my_num_bytes);

      auto src = cudf::io::source_info(file_paths[i]);
      auto opt = cudf::io::csv_reader_options::builder(src);
      if (file_bytes_offset != 0 || !read_header) {
        // Reading the header makes only sense at the start of a file
        // TODO: If the header is read, could sanity check columns for multiple files.
        opt.header(-1);
      }
      opt.delimiter(delimiter);
      opt.na_filter(na_filter);
      opt.dtypes(dtypes_map);
      opt.byte_range_offset(file_bytes_offset);
      opt.byte_range_size(file_bytes_to_read);
      opt.use_cols_indexes(use_cols_indexes);
      opt.names(column_names);

      auto read_table = cudf::io::read_csv(opt, context.get_task_stream(), &mr).tbl;

      // Only add if we read something (otherwise number of cols may be off)
      if (read_table->num_rows() != 0) { tables.emplace_back(std::move(read_table)); }

      // Reading may read additional bytes at the end and less at the start
      // However, there is no need to worry about the actual bytes read,
      // we only worry how much we try to read from the next file.
      my_num_bytes -= file_bytes_to_read;
      my_bytes_offset += file_bytes_to_read;
      total_bytes_seen += file_bytes;
    }

    // Concatenate tables and move the result to the output table
    if (tables.size() == 0) {
      tbl_arg.bind_empty_data();
    } else if (tables.size() == 1) {
      tbl_arg.move_into(std::move(tables.back()), mr);
    } else {
      std::vector<cudf::table_view> table_views;
      for (const auto& table : tables) {
        table_views.push_back(table->view());
      }
      tbl_arg.move_into(cudf::concatenate(table_views, context.get_task_stream(), mr), mr);
    }
  }
};

}  // namespace legate::dataframe::task

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::CSVWrite::register_variants();
  legate::dataframe::task::CSVRead::register_variants();
  return 0;
}();

}  // namespace

namespace legate::dataframe {

void csv_write(LogicalTable& tbl, const std::string& dirpath, char delimiter)
{
  std::filesystem::create_directories(dirpath);
  if (!std::filesystem::is_empty(dirpath)) {
    throw std::invalid_argument("if path exist, it must be an empty directory");
  }
  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::CSVWrite::TASK_CONFIG.task_id());
  argument::add_next_scalar(task, dirpath);
  argument::add_next_scalar_vector(task, tbl.get_column_name_vector());
  argument::add_next_input(task, tbl);
  // legate doesn't accept char so we use int32_t instead
  argument::add_next_scalar(task, static_cast<int32_t>(delimiter));
  runtime->submit(std::move(task));
}

LogicalTable csv_read(const std::string& glob_string,
                      const std::vector<cudf::data_type>& dtypes,
                      bool na_filter,
                      char delimiter,
                      const std::optional<std::vector<std::string>>& names,
                      const std::optional<std::vector<int>>& usecols)
{
  std::vector<std::string> file_paths = parse_glob(glob_string);
  if (file_paths.empty()) { throw std::invalid_argument("no csv files specified"); }

  if (usecols.has_value()) {
    if (!names.has_value()) {
      throw std::invalid_argument("If usecols is given names must also be given.");
    }
    if (usecols.value().size() != dtypes.size()) {
      throw std::invalid_argument("usecols, names, and dtypes must have same number of entries.");
    }
  }
  if (names.has_value() && names.value().size() != dtypes.size()) {
    throw std::invalid_argument("usecols, names, and dtypes must have same number of entries.");
  }

  // We read the column names from the first file.
  // At the moment users must pass in dtypes, otherwise one could read more
  // rows to make a guess (but especially with nullable data that can fail).
  auto source  = cudf::io::source_info(file_paths[0]);
  auto options = cudf::io::csv_reader_options::builder(source);
  if (usecols.has_value()) {
    options.header(-1);
    options.nrows(1);  // Try to read one row to error on a bad file.
  } else {
    options.nrows(0);
  }
  options.delimiter(delimiter);
  auto result = cudf::io::read_csv(options);

  // To use byte-ranges without header parsing, usecols requires all column
  // names to be set.
  std::vector<std::string> all_column_names;
  all_column_names.reserve(result.metadata.schema_info.size());
  for (const auto& column_name_info : result.metadata.schema_info) {
    all_column_names.push_back(column_name_info.name);
  }

  // Get the column names, columns (with dtype), and the column indices.
  // If the user provided names we need to translate those to indices.
  std::vector<std::string> column_names;
  std::vector<LogicalColumn> columns;
  std::vector<int> use_cols_indexes;
  column_names.reserve(dtypes.size());
  columns.reserve(dtypes.size());
  use_cols_indexes.reserve(dtypes.size());
  if (usecols.has_value()) {
    // We seem to have to sort usecols and names?  Just do so with a map...
    std::map<size_t, size_t> reorder_map;
    for (size_t i = 0; i < usecols.value().size(); i++) {
      reorder_map[usecols.value().at(i)] = i;
    }
    for (const auto& [column_index, i] : reorder_map) {
      use_cols_indexes.push_back(column_index);
      column_names.push_back(names.value().at(i));
      columns.emplace_back(LogicalColumn::empty_like(dtypes.at(i), true));
    }
  } else if (!names.has_value()) {
    if (all_column_names.size() != dtypes.size()) {
      throw std::invalid_argument("number of columns in csv doesn't match number of dtypes.");
    }

    for (const auto& name : all_column_names) {
      auto column_index = use_cols_indexes.size();
      column_names.push_back(name);
      columns.emplace_back(LogicalColumn::empty_like(dtypes.at(column_index), true));
      use_cols_indexes.push_back(column_index);
    }
  } else {
    // Translate provided names to (sorted) indexes for passing to the task
    // and create the corresponding columns in the same order.
    // TODO: This can probably be simplified, as we could pass the names and sorting
    //       should be unnecessary.  It is good to check for columns not found, though.
    std::map<std::string, size_t> provided_names;
    for (const auto& name : names.value()) {
      const auto [_, success] = provided_names.insert({name, provided_names.size()});
      if (!success) { throw std::invalid_argument("all column names must be unique"); }
    }

    int column_index = 0;
    for (const auto& column_name_info : result.metadata.schema_info) {
      auto provided = provided_names.find(column_name_info.name);
      if (provided != provided_names.end()) {
        column_names.push_back(column_name_info.name);
        columns.emplace_back(LogicalColumn::empty_like(dtypes[provided->second], true));
        use_cols_indexes.push_back(column_index);
        provided_names.erase(provided);
      }
      column_index++;
    }
    if (provided_names.size() != 0) {
      throw std::invalid_argument("column '" + provided_names.begin()->first +
                                  "' not found in file.");
    }
  }

  LogicalTable ret(std::move(columns), column_names);

  // Get the number of bytes in each file:
  std::vector<size_t> nbytes;
  size_t nbytes_total = 0;
  nbytes.reserve(file_paths.size());
  for (const auto& path : file_paths) {
    auto file_size = std::filesystem::file_size(path);
    nbytes.push_back(file_size);
    nbytes_total += file_size;
  }

  auto runtime          = legate::Runtime::get_runtime();
  legate::AutoTask task = runtime->create_task(get_library(), task::CSVRead::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, file_paths);
  argument::add_next_scalar_vector(task, all_column_names);
  argument::add_next_scalar_vector(task, use_cols_indexes);
  argument::add_next_scalar(task, na_filter);
  // legate doesn't accept char so we use int32_t instead
  argument::add_next_scalar(task, static_cast<int32_t>(delimiter));
  argument::add_next_scalar_vector(task, nbytes);
  argument::add_next_scalar(task, nbytes_total);
  argument::add_next_scalar(task, !usecols.has_value());
  argument::add_next_output(task, ret);
  argument::add_parallel_launch_task(task);
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
