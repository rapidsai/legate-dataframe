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

#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>

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
#include <legate_dataframe/utils.hpp>

#include <legate_dataframe/parquet.hpp>

namespace legate::dataframe::task {

class ParquetWrite : public Task<ParquetWrite, OpCode::ParquetWrite> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetWrite}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const std::string dirpath  = argument::get_next_scalar<std::string>(ctx);
    const auto column_names    = argument::get_next_scalar_vector<std::string>(ctx);
    const auto table           = argument::get_next_input<PhysicalTable>(ctx);
    const std::string filepath = dirpath + "/part." + std::to_string(ctx.rank) + ".parquet";
    auto outfile               = ARROW_RESULT(arrow::io::FileOutputStream::Open(filepath));
    auto props                 = parquet::WriterProperties::Builder().build();
    auto arrow_props           = parquet::ArrowWriterProperties::Builder().build();

    // TODO: memory pool should come from legate
    auto status = parquet::arrow::WriteTable(*table.arrow_table_view(column_names),
                                             arrow::default_memory_pool(),
                                             outfile,
                                             parquet::DEFAULT_MAX_ROW_GROUP_LENGTH,
                                             props,
                                             arrow_props);
  }

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
    cudf::io::write_parquet(options, ctx.stream());
  }
};

namespace {

std::pair<std::vector<std::string>, std::vector<std::vector<int>>> find_files_and_row_groups(
  const std::vector<std::string>& file_paths,
  const std::vector<size_t>& ngroups_per_file,
  size_t from_row_group,
  size_t num_row_groups)
{
  // Iterate through the file and nrow list and read as many rows from the
  // files as this rank should read while skipping those of the other tasks.
  size_t total_groups_seen = 0;  // offset into the current file

  std::vector<std::string> files;
  std::vector<std::vector<int>> row_groups;
  for (size_t i = 0; i < file_paths.size(); i++) {
    size_t file_groups = ngroups_per_file[i];

    if (num_row_groups == 0) {
      break;
    } else if (from_row_group >= file_groups) {
      from_row_group -= file_groups;  // full file is before our first row group
      continue;
    }

    files.push_back(file_paths[i]);

    size_t num_groups_from_this_file = std::min(num_row_groups, file_groups - from_row_group);
    auto file_row_groups             = std::vector<int>(num_groups_from_this_file);
    std::iota(file_row_groups.begin(), file_row_groups.end(), from_row_group);
    row_groups.push_back(std::move(file_row_groups));

    num_row_groups -= num_groups_from_this_file;
  }

  return {std::move(files), std::move(row_groups)};
}

/*
 * Helper since create_output_buffer needs to be typed, but we dispatch again later
 * so a `void *` return is OK.
 */
struct create_result_store_fn {
  template <legate::Type::Code CODE>
  void* operator()(const legate::PhysicalStore& store, const legate::Point<2>& shape)
  {
    using VAL = legate::type_of<CODE>;
    auto buf  = store.create_output_buffer<VAL, 2>(shape, true);
    return buf.ptr({0, 0});
  }
};

}  // namespace

// Arrow wants column indices instead of names
// Find the columns
std::vector<int> GetArrowColumnIndices(const std::string& file_path,
                                       const std::vector<std::string>& columns)
{
  // Open the file and get the schema
  auto input = ARROW_RESULT(arrow::io::ReadableFile::Open(file_path));
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  auto status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
  std::shared_ptr<arrow::Schema> schema;
  status = arrow_reader->GetSchema(&schema);

  // Find the column indices
  std::vector<int> column_indices;
  column_indices.reserve(columns.size());
  for (const auto& col : columns) {
    auto idx = schema->GetFieldIndex(col);
    if (idx < 0) { throw std::runtime_error("Column " + col + " not found in Parquet schema."); }
    column_indices.push_back(idx);
  }
  return column_indices;
}

class ParquetRead : public Task<ParquetRead, OpCode::ParquetRead> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetRead}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
    const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
    const auto nrow_groups_total = argument::get_next_scalar<size_t>(ctx);
    PhysicalTable tbl_arg        = argument::get_next_output<PhysicalTable>(ctx);
    argument::get_parallel_launch_task(ctx);

    if (file_paths.size() != ngroups_per_file.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }

    auto [my_groups_offset, my_num_groups] =
      evenly_partition_work(nrow_groups_total, ctx.rank, ctx.nranks);

    if (my_num_groups == 0) {
      tbl_arg.bind_empty_data();
      return;
    }

    auto column_indices = GetArrowColumnIndices(file_paths.at(0), columns);
    // Iterate over files
    auto [files, row_groups] =
      find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (int i = 0; i < files.size(); i++) {
      auto input = ARROW_RESULT(arrow::io::ReadableFile::Open(files[i]));
      std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
      auto status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
      std::unique_ptr<arrow::RecordBatchReader> batch_reader;
      status = arrow_reader->GetRecordBatchReader(row_groups[i], column_indices, &batch_reader);
      tables.push_back(ARROW_RESULT(batch_reader->ToTable()));
    }
    // Concatenate the tables
    if (tables.size() == 0) {
      tbl_arg.bind_empty_data();
    } else {
      tbl_arg.move_into(std::move(ARROW_RESULT(arrow::ConcatenateTables(tables))));
    }
  }

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
    const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
    const auto nrow_groups_total = argument::get_next_scalar<size_t>(ctx);
    PhysicalTable tbl_arg        = argument::get_next_output<PhysicalTable>(ctx);
    argument::get_parallel_launch_task(ctx);

    if (file_paths.size() != ngroups_per_file.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }

    auto [my_groups_offset, my_num_groups] =
      evenly_partition_work(nrow_groups_total, ctx.rank, ctx.nranks);

    if (my_num_groups == 0) {
      tbl_arg.bind_empty_data();
      return;
    }

    auto [files, row_groups] =
      find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);

    auto src = cudf::io::source_info(files);
    auto opt = cudf::io::parquet_reader_options::builder(src);
    opt.columns(columns);
    opt.row_groups(row_groups);
    auto res = cudf::io::read_parquet(opt, ctx.stream(), ctx.mr()).tbl;

    tbl_arg.move_into(std::move(res));
  }
};

class ParquetReadArray : public Task<ParquetReadArray, OpCode::ParquetReadArray> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ParquetReadArray}};

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
    const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
    const auto row_group_ranges  = argument::get_next_scalar_vector<legate::Rect<2>>(ctx);
    const auto nrow_groups_total = argument::get_next_scalar<size_t>(ctx);
    auto null_value              = ctx.get_next_scalar_arg();
    auto out                     = ctx.get_next_output_arg();
    argument::get_parallel_launch_task(ctx);

    auto [my_groups_offset, my_num_groups] =
      evenly_partition_work(nrow_groups_total, ctx.rank, ctx.nranks);

    if (file_paths.size() != ngroups_per_file.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }
    if (my_num_groups == 0) {
      out.data().bind_empty_data();
      if (out.nullable()) { out.null_mask().bind_empty_data(); }
      return;
    }

    const size_t ncols = columns.size();

    legate::Rect<2> start = row_group_ranges.at(my_groups_offset);
    legate::Rect<2> end   = row_group_ranges.at(my_groups_offset + my_num_groups - 1);

    auto num_output_rows = end.hi[0] - start.lo[0] + 1;
    void* data_ptr       = legate::type_dispatch(out.data().code(),
                                           create_result_store_fn{},
                                           out.data(),
                                           legate::Point<2>({num_output_rows, ncols}));
    std::optional<bool*> null_ptr;
    if (out.nullable()) {
      auto null_buf = out.null_mask().create_output_buffer<bool, 2>(
        legate::Point<2>({num_output_rows, ncols}), true);
      auto ptr = null_buf.ptr({0, 0});
      null_ptr = ptr;
    }

    if (columns.size() != start.hi[1] - start.lo[1] + 1) {
      throw std::runtime_error("internal error: columns size and result shape mismatch");
    }

    auto column_indices = GetArrowColumnIndices(file_paths.at(0), columns);
    // Iterate over files
    auto [files, row_groups] =
      find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);
    size_t rows_already_written = 0;
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (int i = 0; i < files.size(); i++) {
      auto input = ARROW_RESULT(arrow::io::ReadableFile::Open(files[i]));
      std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
      auto status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
      std::unique_ptr<arrow::RecordBatchReader> batch_reader;
      status     = arrow_reader->GetRecordBatchReader(row_groups[i], column_indices, &batch_reader);
      auto table = ARROW_RESULT(batch_reader->ToTable());

      if (end.hi[0] - start.lo[0] + 1 < rows_already_written + table->num_rows()) {
        throw std::runtime_error("internal error: output smaller than expected.");
      }

      // Write to output array, this is a transposed copy.
      copy_into_tranposed(ctx, data_ptr, null_ptr, table, null_value, out.data().type());

      if (null_ptr.has_value()) { null_ptr = null_ptr.value() + table->num_rows() * ncols; }
      data_ptr =
        static_cast<char*>(data_ptr) + table->num_rows() * ncols * out.data().type().size();
      rows_already_written += table->num_rows();
    }
  }
  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
    const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
    const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
    const auto row_group_ranges  = argument::get_next_scalar_vector<legate::Rect<2>>(ctx);
    const auto nrow_groups_total = argument::get_next_scalar<size_t>(ctx);
    auto null_value              = ctx.get_next_scalar_arg();
    auto out                     = ctx.get_next_output_arg();
    argument::get_parallel_launch_task(ctx);

    auto expected_type_id = to_cudf_type_id(out.type().code());

    auto [my_groups_offset, my_num_groups] =
      evenly_partition_work(nrow_groups_total, ctx.rank, ctx.nranks);

    if (file_paths.size() != ngroups_per_file.size()) {
      throw std::runtime_error("internal error: file path and nrows size mismatch");
    }
    if (my_num_groups == 0) {
      out.data().bind_empty_data();
      if (out.nullable()) { out.null_mask().bind_empty_data(); }
      return;
    }

    const size_t ncols = columns.size();

    // TODO: This is hack (including the partitioning above).  We should be passing in a bound
    // output with image constraints on row_group_ranges at which point we would just need to know
    // the row groups assigned to us (the number of rows will be correct in the output array shape).
    legate::Rect<2> start = row_group_ranges.at(my_groups_offset);
    legate::Rect<2> end   = row_group_ranges.at(my_groups_offset + my_num_groups - 1);

    void* data_ptr = legate::type_dispatch(out.data().code(),
                                           create_result_store_fn{},
                                           out.data(),
                                           legate::Point<2>({end.hi[0] - start.lo[0] + 1, ncols}));
    std::optional<bool*> null_ptr;
    if (out.nullable()) {
      auto null_buf = out.null_mask().create_output_buffer<bool, 2>(
        legate::Point<2>({end.hi[0] - start.lo[0] + 1, ncols}), true);
      auto ptr = null_buf.ptr({0, 0});
      null_ptr = ptr;
    }

    if (columns.size() != start.hi[1] - start.lo[1] + 1) {
      throw std::runtime_error("internal error: columns size and result shape mismatch");
    }

    auto [files, row_groups] =
      find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);

    // Read a few hundred MiB at a time (assumes datatype isn't very narrow).
    // (The actual decompression etc. will be done in larger chunks.)
    auto chunksize = ((1 << 25) + ncols - 1) / ncols;

    auto src = cudf::io::source_info(files);
    auto opt = cudf::io::parquet_reader_options::builder(src);
    opt.columns(columns);
    opt.row_groups(row_groups);

    auto reader =
      cudf::io::chunked_parquet_reader(chunksize, chunksize, opt, ctx.stream(), ctx.mr());
    size_t rows_already_written = 0;
    while (reader.has_next()) {
      auto tbl = reader.read_chunk().tbl;
      /* Check if all columns are of the right type and cast them if not. */
      auto column_vec = tbl->release();
      for (auto& col : column_vec) {
        if (col->type().id() != expected_type_id) {
          col = cudf::cast(col->view(), cudf::data_type{expected_type_id}, ctx.stream(), ctx.mr());
        }
      }
      auto cast_tbl = cudf::table(std::move(column_vec));

      if (end.hi[0] - start.lo[0] + 1 < rows_already_written + cast_tbl.num_rows()) {
        throw std::runtime_error("internal error: output smaller than expected.");
      }
      // Write to output array, this is a transposed copy.
      copy_into_tranposed(ctx, data_ptr, null_ptr, cast_tbl.view(), null_value);

      if (null_ptr.has_value()) { null_ptr = null_ptr.value() + cast_tbl.num_rows() * ncols; }
      data_ptr =
        static_cast<char*>(data_ptr) + cast_tbl.num_rows() * ncols * out.data().type().size();
      rows_already_written += cast_tbl.num_rows();
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
  std::vector<cudf::data_type> column_types;
  std::vector<bool> column_nullable;
  std::vector<size_t> nrow_groups;
  std::vector<legate::Rect<2>> row_group_ranges_vec;
  LogicalArray row_group_ranges;
  size_t nrow_groups_total;
  size_t nrows_total;
};

ParquetReadInfo get_parquet_info(const std::string& glob_string,
                                 const std::optional<std::vector<std::string>>& columns)
{
  std::vector<std::string> file_paths = parse_glob(glob_string);
  if (file_paths.empty()) { throw std::invalid_argument("no parquet files specified"); }

  // TODO: Using the default memory pool, hopefully it doesn't matter anyway here.
  auto pool = arrow::default_memory_pool();

  // Open the first file to get schema information
  auto reader = ARROW_RESULT(arrow::io::ReadableFile::Open(file_paths[0]));

  // Newer versions arrow have versions that return a result which is more convenient (also below)
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  auto status = parquet::arrow::OpenFile(reader, pool, &parquet_reader);
  if (!status.ok()) {
    throw std::runtime_error("failed to open parquet file: " + status.ToString());
  }
  std::shared_ptr<arrow::Schema> schema;
  status = parquet_reader->GetSchema(&schema);
  if (!status.ok()) { throw std::runtime_error("failed to get schema: " + status.ToString()); }

  // Get the column metadata from the schema (depends on whether columns are specified)
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  std::vector<bool> column_nullable;
  if (!columns.has_value()) {
    column_names.reserve(schema->num_fields());
    column_types.reserve(schema->num_fields());
    column_nullable.reserve(schema->num_fields());
    for (int i = 0; i < schema->num_fields(); i++) {
      auto name = schema->field(i)->name();
      column_names.emplace_back(name);
      auto arrow_type = schema->field(i)->type();
      column_types.emplace_back(to_cudf_type(*arrow_type.get()));
      column_nullable.emplace_back(schema->field(i)->nullable());
    }
  } else {
    std::map<std::string, int> name_to_index;
    for (int i = 0; i < schema->num_fields(); i++) {
      name_to_index[schema->field(i)->name()] = i;
    }
    column_names.reserve(columns.value().size());
    column_types.reserve(columns.value().size());
    column_nullable.reserve(columns.value().size());
    for (auto& name : columns.value()) {
      // Validate column names
      if (name_to_index.count(name) == 0) {
        throw std::invalid_argument("column was not found in parquet file: " + std::string(name));
      }
      auto i = name_to_index.at(name);
      column_names.emplace_back(name);
      auto arrow_type = schema->field(i)->type();
      column_types.emplace_back(to_cudf_type(*arrow_type.get()));
      column_nullable.emplace_back(schema->field(i)->nullable());
    }
  }

  // We read by row groups, because this is how the decompression works.
  // If the row groups are huge, that may not be ideal, but there is not much
  // we can do about it at the moment.
  // Reading in chunks smaller than row groups is just not well supported.
  size_t nrows_total       = 0;
  size_t nrow_groups_total = 0;
  std::vector<size_t> nrow_groups;
  std::vector<legate::Rect<2>> row_group_ranges;
  for (const auto& path : file_paths) {
    auto reader         = ARROW_RESULT(arrow::io::ReadableFile::Open(path));
    auto parquet_reader = parquet::ParquetFileReader::Open(reader);
    auto metadata       = parquet_reader->metadata();

    nrow_groups.push_back(metadata->num_row_groups());
    nrow_groups_total += metadata->num_row_groups();
    for (int i = 0; i < metadata->num_row_groups(); i++) {
      auto row_group          = parquet_reader->RowGroup(i);
      auto row_group_metadata = row_group->metadata();

      auto nrows_in_group = row_group_metadata->num_rows();
      // TODO: Legate limitations force us to use a 2D rect here, which we don't actually want/need
      // but the array is 2-D and the broadcast constraint currently doesn't work to make it 1-D
      // for this purpose. (As of legate 25.05)
      row_group_ranges.emplace_back(legate::Rect<2>(
        {nrows_total, 0}, {nrows_total + nrows_in_group - 1, column_names.size() - 1}));
      nrows_total += nrows_in_group;
    }
  }

  auto runtime = legate::Runtime::get_runtime();
  auto row_group_ranges_arr =
    runtime->create_array({row_group_ranges.size()}, legate::rect_type(2));
  auto ptr = row_group_ranges_arr.get_physical_array()
               .data()
               .write_accessor<legate::Rect<2>, 1, false>()
               .ptr(0);
  std::copy(row_group_ranges.begin(), row_group_ranges.end(), ptr);

  return {std::move(file_paths),
          std::move(column_names),
          std::move(column_types),
          std::move(column_nullable),
          std::move(nrow_groups),
          std::move(row_group_ranges),
          row_group_ranges_arr,
          nrow_groups_total,
          nrows_total};
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
  auto info = get_parquet_info(glob_string, columns);

  std::vector<LogicalColumn> logical_columns;
  logical_columns.reserve(info.column_types.size());
  for (int i = 0; i < info.column_types.size(); i++) {
    logical_columns.emplace_back(
      LogicalColumn::empty_like(info.column_types.at(i), info.column_nullable.at(i), false));
  }
  auto ret = LogicalTable(std::move(logical_columns), info.column_names);

  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetRead::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, info.file_paths);
  argument::add_next_scalar_vector(task, info.column_names);
  argument::add_next_scalar_vector(task, info.nrow_groups);
  argument::add_next_scalar(task, info.nrow_groups_total);
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
    auto cudf_type = info.column_types.at(0);
    if (!cudf::is_numeric(cudf_type)) {
      throw std::invalid_argument("only numeric columns are supported for parquet_read_array");
    }
    for (auto& type : info.column_types) {
      if (type.id() != type.id()) {
        throw std::invalid_argument("all columns must have the same type");
      }
    }

    legate_type = to_legate_type(cudf_type.id());
  } else {
    legate_type = type.value();

    auto cudf_type = cudf::data_type{to_cudf_type_id(legate_type.code())};
    for (auto& type : info.column_types) {
      if (!cudf::is_supported_cast(type, cudf_type)) {
        throw std::invalid_argument("Cannot cast all columns to specified type");
      }
    }
  }

  // If all columns are not nullable, we don't have to worry about null values
  auto nullable = std::any_of(info.column_nullable.begin(),
                              info.column_nullable.end(),
                              [](bool nullable) { return nullable; });
  if (nullable) {
    // Otherwise, see if we fill the null values
    nullable = null_value.type().code() == Type::Code::NIL;
    if (!nullable) {
      if (null_value.type() != legate_type) {
        throw std::invalid_argument("null value must be null or have the same type as the result");
      }
    }
  }

  // See below, this should not be late-bound, but that requires working imagine constraints.
  auto ret = runtime->create_array(legate_type, 2, nullable);

  legate::AutoTask task =
    runtime->create_task(get_library(), task::ParquetReadArray::TASK_CONFIG.task_id());
  argument::add_next_scalar_vector(task, info.file_paths);
  argument::add_next_scalar_vector(task, info.column_names);
  argument::add_next_scalar_vector(task, info.nrow_groups);
  argument::add_next_scalar_vector(task, info.row_group_ranges_vec);
  argument::add_next_scalar(task, info.nrow_groups_total);
  argument::add_next_scalar(task, null_value);

  // auto constraint_var = task.add_input(info.row_group_ranges);
  auto var = task.add_output(ret);
  task.add_constraint(legate::broadcast(var, {1}));
  argument::add_parallel_launch_task_2d(task);
  // See issue 2398
  // task.add_constraint(legate::image(constraint_var, var));
  runtime->submit(std::move(task));
  return ret;
}

}  // namespace legate::dataframe
