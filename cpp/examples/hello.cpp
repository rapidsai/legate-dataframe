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

#include <legate.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/filling.hpp>
#include <legate_dataframe/parquet.hpp>
#include <legate_dataframe/unaryop.hpp>
#include <legate_dataframe/utils.hpp>

int main(void)
{
  // First we initialize Legate use either `legate` or `LEGATE_CONFIG` to customize launch
  legate::start();

  // Then let's create a new logical column
  legate::dataframe::LogicalColumn col_a = legate::dataframe::sequence(20, -10);

  // Compute the absolute value of each row in `col_a`
  legate::dataframe::LogicalColumn col_b = unary_operation(col_a, "abs");

  // Create a new logical table that contains the two existing columns (zero-copy)
  // naming them "a" and "b"
  legate::dataframe::LogicalTable tbl_a({col_a, col_b}, std::vector<std::string>({"a", "b"}));

  // We can write the logical table to disk using the Parquet file format.
  // The table is written into multiple files, one file per partition:
  //      /tmpdir/
  //          ├── part-0.parquet
  //          ├── part-1.parquet
  //          ├── part-2.parquet
  //          └── ...
  legate::dataframe::parquet_write(tbl_a, "./my_parquet_file");

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  // Then we can read the parquet files back into a logical table. We
  // provide a Glob string that reference all the parquet files that
  // should go into the logical table.
  auto files = legate::dataframe::parse_glob("./my_parquet_file/*.parquet");
  auto tbl_b = legate::dataframe::parquet_read(files);

  // Clean up
  std::filesystem::remove_all("./my_parquet_file");
  return 0;
}
