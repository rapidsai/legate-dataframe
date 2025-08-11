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
#include <fstream>
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

#include <legate_dataframe/csv.hpp>

namespace legate::dataframe::task {

/*static*/ void CSVWrite::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const std::string dirpath  = argument::get_next_scalar<std::string>(ctx);
  const auto column_names    = argument::get_next_scalar_vector<std::string>(ctx);
  const auto tbl             = argument::get_next_input<PhysicalTable>(ctx);
  const std::string filepath = dirpath + "/part." + std::to_string(ctx.rank) + ".csv";
  const auto delimiter       = static_cast<char>(argument::get_next_scalar<int32_t>(ctx));

  auto dest    = cudf::io::sink_info(filepath);
  auto options = cudf::io::csv_writer_options::builder(dest, tbl.table_view());
  options.names(column_names);
  options.inter_column_delimiter(delimiter);

  cudf::io::write_csv(options, ctx.stream());
}

/* static */ void CSVRead::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto file_paths       = argument::get_next_scalar_vector<std::string>(ctx);
  const auto column_names     = argument::get_next_scalar_vector<std::string>(ctx);
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

  auto [my_bytes_offset, my_num_bytes] = evenly_partition_work(nbytes_total, ctx.rank, ctx.nranks);

  auto dtypes = tbl_arg.cudf_types();

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

    auto read_table = cudf::io::read_csv(opt, ctx.stream(), ctx.mr()).tbl;

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
    tbl_arg.move_into(std::move(tables.back()));
  } else {
    std::vector<cudf::table_view> table_views;
    for (const auto& table : tables) {
      table_views.push_back(table->view());
    }
    tbl_arg.move_into(cudf::concatenate(table_views, ctx.stream(), ctx.mr()));
  }
}

}  // namespace legate::dataframe::task
