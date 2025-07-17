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

#include <string>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Write table to Parquet files.
 *
 * Each partition will be written to a separate file.
 *
 * Files will be created in the specified output directory using the convention ``part.0.parquet``,
 * ``part.1.parquet``, ``part.2.parquet``, ... and so on for each partition in the table:
 *
 *      /path/to/output/
 *          ├── part-0.parquet
 *          ├── part-1.parquet
 *          ├── part-2.parquet
 *          └── ...
 *
 * @param tbl The table to write.
 * @param path Destination directory for data.
 */
void parquet_write(LogicalTable& tbl, const std::string& dirpath);

/**
 * @brief Read Parquet files into a LogicalTable
 *
 * Files are currently read into N partitions where N is the number of GPU workers used.
 * The partitions are split by row-group, meaning that each reads approximately the
 * same number of row-groups (possibly over multiple files).
 * If the number of row-groups does not split evenly, the first partitions will
 * contain one additional row-group.
 *
 * Note that file order is currently glob/string sorted.
 *
 * @param files The parquet files to read.
 * @param columns The columns names to read.
 * @return The read LogicalTable
 */
LogicalTable parquet_read(const std::vector<std::string>& files,
                          const std::optional<std::vector<std::string>>& columns = std::nullopt);

/**
 * @brief Read Parquet files into a legate Array
 *
 * Note that file order is currently glob/string sorted.  All columns that are being read must have
 * the same type and be compatible with a legate type, currently only numeric types are supported.
 *
 * @param files The parquet files to read.
 * @param columns The columns names to read.
 * @param nullable If set to ``False``, assume that the file does contain any nulls.
 * @return The read LogicalTable
 */
legate::LogicalArray parquet_read_array(const std::vector<std::string>& files,
                                        const std::optional<std::vector<std::string>>& columns,
                                        const legate::Scalar& null_value,
                                        const std::optional<legate::Type>& type);

}  // namespace legate::dataframe
