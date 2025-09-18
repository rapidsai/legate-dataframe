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

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/unary.hpp>
#include <legate_dataframe/core/transposed_copy.h>

#include <legate_dataframe/parquet.hpp>

namespace legate::dataframe::task {

/*static*/ void ParquetWrite::gpu_variant(legate::TaskContext context)
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

/*static*/ void ParquetRead::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
  const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
  const auto column_indices    = argument::get_next_scalar_vector<int>(ctx);  // Unused by cudf
  const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
  const auto nrow_groups_total = argument::get_next_scalar<size_t>(ctx);
  PhysicalTable tbl_arg        = argument::get_next_output<PhysicalTable>(ctx);

  size_t my_groups_offset{};
  size_t my_num_groups{};

  if (!get_prefer_eager_allocations()) {
    /* Infer ranges from number of ranks */
    argument::get_parallel_launch_task(ctx);
    std::tie(my_groups_offset, my_num_groups) =
      evenly_partition_work(nrow_groups_total, ctx.rank, ctx.nranks);
  } else {
    /* Infer ranges from constraint array assigned to us */
    auto row_group_ranges = ctx.get_next_input_arg();
    my_groups_offset      = row_group_ranges.shape<1>().lo[0];
    my_num_groups         = row_group_ranges.shape<1>().hi[0] - my_groups_offset + 1;
  }

  if (file_paths.size() != ngroups_per_file.size()) {
    throw std::runtime_error("internal error: file path and nrows size mismatch");
  }

  if (my_num_groups == 0) {
    if (!get_prefer_eager_allocations()) { tbl_arg.bind_empty_data(); }
    return;
  }

  auto [files, row_groups] =
    find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);

  auto src = cudf::io::source_info(files);
  auto opt = cudf::io::parquet_reader_options::builder(src);
  opt.columns(columns);
  opt.row_groups(row_groups);
  // If pandas metadata is read, libcudf may read index columns without this.
  opt.use_pandas_metadata(false);
  auto res = cudf::io::read_parquet(opt, ctx.stream(), ctx.mr()).tbl;

  if (get_prefer_eager_allocations()) {
    tbl_arg.copy_into(std::move(res));
  } else {
    tbl_arg.move_into(std::move(res));
  }
}

/*static*/ void ParquetReadByRows::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto file_paths       = argument::get_next_scalar_vector<std::string>(ctx);
  const auto columns          = argument::get_next_scalar_vector<std::string>(ctx);
  const auto column_indices   = argument::get_next_scalar_vector<int>(ctx);  // Unused by cudf
  const auto ngroups_per_file = argument::get_next_scalar_vector<size_t>(ctx);
  const auto nrows_per_group  = argument::get_next_scalar_vector<size_t>(ctx);
  const auto nrows_total      = argument::get_next_scalar<size_t>(ctx);
  PhysicalTable tbl_arg       = argument::get_next_output<PhysicalTable>(ctx);

  size_t my_row_offset{};
  size_t my_nrows{};

  if (!get_prefer_eager_allocations()) {
    /* Infer ranges from number of ranks */
    argument::get_parallel_launch_task(ctx);
    std::tie(my_row_offset, my_nrows) = evenly_partition_work(nrows_total, ctx.rank, ctx.nranks);
  } else {
    /* Infer ranges from constraint array assigned to us */
    my_row_offset = tbl_arg.global_row_offset();
    my_nrows      = tbl_arg.num_rows();
  }

  if (my_nrows == 0) {
    if (!get_prefer_eager_allocations()) { tbl_arg.bind_empty_data(); }
    return;
  }

  if (file_paths.size() != ngroups_per_file.size()) {
    throw std::runtime_error("internal error: file path and nrows size mismatch");
  }

  /*
   * Remove initial files that we definitely do not need, this avoids touching them
   * and also works around a possible issue with libcudf 25.06 that is fixed in 25.08.
   * (could pass the actual rows in the file in to simplify this.)
   */
  size_t skip_files   = 0;
  size_t ngroups_seen = 0;
  for (auto ngroups : ngroups_per_file) {
    size_t file_rows = 0;
    for (size_t group = 0; group < ngroups; group++) {
      file_rows += nrows_per_group[ngroups_seen + group];
    }
    ngroups_seen += ngroups;
    if (file_rows > my_row_offset) {
      break;  // This file is used, we are done.
    }
    skip_files++;
    my_row_offset -= file_rows;
  }
  auto files = std::vector<std::string>(file_paths.begin() + skip_files, file_paths.end());

  // In principle, we know the exact row groups already, but the cudf reader supports row skipping
  // directly, so hope that is better.  It is not possible to combine both (as of 25.06 at least).
  // (We could also use a chunked reader on 25.08+.)
  auto src = cudf::io::source_info(files);
  auto opt = cudf::io::parquet_reader_options::builder(src);
  opt.columns(columns);
  opt.skip_rows(my_row_offset);
  opt.num_rows(my_nrows);
  // If pandas metadata is read, libcudf may read index columns without this.
  opt.use_pandas_metadata(false);
  auto res = cudf::io::read_parquet(opt, ctx.stream(), ctx.mr()).tbl;

  if (res->num_rows() == 0) {
    if (!get_prefer_eager_allocations()) { tbl_arg.bind_empty_data(); }
    return;
  }

  if (get_prefer_eager_allocations()) {
    tbl_arg.copy_into(std::move(res));
  } else {
    tbl_arg.move_into(std::move(res));
  }
}

/*static*/ void ParquetReadArray::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto file_paths        = argument::get_next_scalar_vector<std::string>(ctx);
  const auto columns           = argument::get_next_scalar_vector<std::string>(ctx);
  const auto column_indices    = argument::get_next_scalar_vector<int>(ctx);  // Unused by cudf
  const auto ngroups_per_file  = argument::get_next_scalar_vector<size_t>(ctx);
  const auto row_group_ranges  = argument::get_next_scalar_vector<legate::Rect<1>>(ctx);
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

  // TODO: This is hack (including the partitioning above).  We should be passing in a bound
  // output with image constraints on row_group_ranges at which point we would just need to know
  // the row groups assigned to us (the number of rows will be correct in the output array shape).
  legate::Rect<1> start = row_group_ranges.at(my_groups_offset);
  legate::Rect<1> end   = row_group_ranges.at(my_groups_offset + my_num_groups - 1);

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

  auto [files, row_groups] =
    find_files_and_row_groups(file_paths, ngroups_per_file, my_groups_offset, my_num_groups);

  // Read a few hundred MiB at a time (actual limit is a multiple due to decompression
  // and that may also just need more memory as well, there may be other components).
  auto chunksize = 500 * 1024 * 1024;

  auto src = cudf::io::source_info(files);
  auto opt = cudf::io::parquet_reader_options::builder(src);
  opt.columns(columns);
  opt.row_groups(row_groups);

  auto reader = cudf::io::chunked_parquet_reader(chunksize, chunksize, opt, ctx.stream(), ctx.mr());
  size_t rows_already_written = 0;
  while (reader.has_next()) {
    auto tbl = reader.read_chunk().tbl;

    if (end.hi[0] - start.lo[0] + 1 < rows_already_written + tbl->num_rows()) {
      throw std::runtime_error("internal error: output smaller than expected.");
    }
    // Write to output array, this is a transposed copy.
    copy_into_tranposed(ctx, data_ptr, null_ptr, tbl->release(), null_value, out.data().type());

    if (null_ptr.has_value()) { null_ptr = null_ptr.value() + tbl->num_rows() * ncols; }
    data_ptr = static_cast<char*>(data_ptr) + tbl->num_rows() * ncols * out.data().type().size();
    rows_already_written += tbl->num_rows();
  }
}

}  // namespace legate::dataframe::task
