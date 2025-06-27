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

#include <legate.h>

#include "test_utils.hpp"
#include <legate_dataframe/parquet.hpp>
#include <legate_dataframe/utils.hpp>

using namespace legate::dataframe;

template <typename T>
struct NumericParquetTest : public testing::Test {};

TYPED_TEST_SUITE(NumericParquetTest, NumericTypes);

TYPED_TEST(NumericParquetTest, ReadWrite)
{
  TempDir tmp_dir;
  auto filepath = tmp_dir.path() / "parquet_file";
  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}));
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}));
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({a, b}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");

  EXPECT_TRUE(tbl_a.get_arrow()->Equals(*tbl_b.get_arrow()));
}

TYPED_TEST(NumericParquetTest, ReadColumnSubset)
{
  TempDir tmp_dir;
  auto filepath = tmp_dir.path() / "parquet_file";
  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}));
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}));
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({a, b}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  std::vector<std::string> columns({"b"});
  tbl_a              = tbl_a.select(columns);
  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet", columns);

  EXPECT_TRUE(tbl_a.get_arrow()->Equals(*tbl_b.get_arrow()));
}

TYPED_TEST(NumericParquetTest, ReadWriteSingleItem)
{
  TempDir tmp_dir;
  auto filepath = tmp_dir.path() / "parquet_file";
  LogicalColumn a(narrow<TypeParam>({1}));
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({a}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");
  EXPECT_TRUE(tbl_a.get_arrow()->Equals(*tbl_b.get_arrow()));
}

TEST(StringsParquetTest, ReadWrite)
{
  TempDir tmp_dir;
  auto filepath = tmp_dir.path() / "parquet_file";

  const std::vector<std::string> column_names({"a"});
  std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
  LogicalTable tbl_a({LogicalColumn{strings}}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");

  EXPECT_TRUE(tbl_a.get_arrow()->Equals(*tbl_b.get_arrow()))
    << "Expected: " << tbl_a.get_arrow()->ToString() << "\n"
    << "Actual: " << tbl_b.get_arrow()->ToString();
}
