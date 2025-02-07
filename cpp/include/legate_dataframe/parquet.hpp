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
 * The partitions are split by row, meaning that each reads approximately the
 * same number of rows (possibly over multiple files).
 * If the number of rows does not split evenly, the first partitions will
 * contain one additional row.
 *
 * Note that file order is currently glob/string sorted.
 *
 * @param glob_string The glob string to specify the Parquet files. All glob matches must be valid
 * Parquet files and have the same LogicalTable data types. See <https://linux.die.net/man/7/glob>.
 * @param columns The columns names to read.
 * @return The read LogicalTable
 */
LogicalTable parquet_read(const std::string& glob_string,
                          const std::optional<std::vector<std::string>>& columns = std::nullopt);

}  // namespace legate::dataframe
