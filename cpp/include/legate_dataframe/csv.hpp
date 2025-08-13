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

#pragma once

#include <string>

#include <legate.h>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

namespace task {

class CSVRead : public Task<CSVRead, OpCode::CSVRead> {
 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

class CSVWrite : public Task<CSVWrite, OpCode::CSVWrite> {
 public:
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}
                                                .with_has_allocations(true)
                                                .with_elide_device_ctx_sync(true)
                                                .with_has_side_effect(true);
  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_has_side_effect(true);
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};
}  // namespace task
/**
 * @brief Write table to csv files.
 *
 * Each partition will be written to a separate file.
 *
 * Files will be created in the specified output directory using the convention ``part.0.csv``,
 * ``part.1.csv``, ``part.2.csv``, ... and so on for each partition in the table:
 *
 *      /path/to/output/
 *          ├── part.0.csv
 *          ├── part.1.csv
 *          ├── part.2.csv
 *          └── ...
 *
 * @param tbl The table to write.
 * @param path Destination directory for data.
 * @param delimiter The field delimiter.
 */
void csv_write(LogicalTable& tbl, const std::string& dirpath, char delimiter = ',');

/**
 * @brief Read csv files into a LogicalTable
 *
 * Files are currently read into N partitions where N is the number of GPU workers used.
 * The partitions are split by row, meaning that each reads approximately the
 * same number of rows (possibly over multiple files).
 * If the number of rows does not split evenly, the first partitions will
 * contain one additional row.
 *
 * Note that file order is currently glob/string sorted.
 *
 * TODO: We should replace some/all params with cudf::io::csv_reader_options eventually.
 *       As if writing, this would be better with pylibcudf 25.02 which cannot be quite used.
 *
 * @param files The csv files to read.
 * @param dtypes The cudf type for each column (must match usecols).
 * @param na_filter Whether to detect missing values, set to false to improve performance.
 * @param delimiter The field delimiter.
 * @param names The column names to read from the file, if not passed reads all columns.
 * @param usecols Column index in file.  If given, assumes the file includes no header.
 * passing `usecols_idx` without names is not supported.
 * @return The read LogicalTable.
 */
LogicalTable csv_read(const std::vector<std::string>& files,
                      const std::vector<std::shared_ptr<arrow::DataType>>& dtypes,
                      bool na_filter                                       = true,
                      char delimiter                                       = ',',
                      const std::optional<std::vector<std::string>>& names = std::nullopt,
                      const std::optional<std::vector<int>>& usecols       = std::nullopt);

}  // namespace legate::dataframe
