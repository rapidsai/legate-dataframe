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

#include <legate_dataframe/parquet.hpp>
#include <legate_dataframe/utils.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

using namespace legate::dataframe;

template <typename T>
struct NumericParquetTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NumericParquetTest, cudf::test::NumericTypes);

TYPED_TEST(NumericParquetTest, ReadWrite)
{
  TempDir tmp_dir{false};
  auto filepath = tmp_dir.path() / "parquet_file";
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TYPED_TEST(NumericParquetTest, ReadColumnSubset)
{
  TempDir tmp_dir{false};
  auto filepath = tmp_dir.path() / "parquet_file";
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  std::vector<std::string> columns({"b"});
  tbl_a              = tbl_a.select(columns);
  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet", columns);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_b.get_column_name_vector() == columns);
}

TYPED_TEST(NumericParquetTest, ReadWriteSingleItem)
{
  TempDir tmp_dir{false};
  auto filepath = tmp_dir.path() / "parquet_file";
  cudf::test::fixed_width_column_wrapper<TypeParam> a({1});
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({LogicalColumn{a}}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TEST(StringsParquetTest, ReadWrite)
{
  TempDir tmp_dir;
  auto filepath = tmp_dir.path() / "parquet_file";

  std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
  cudf::test::strings_column_wrapper s(strings.begin(), strings.end());
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({LogicalColumn{s}}, column_names);

  parquet_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  LogicalTable tbl_b = parquet_read(tmp_dir.path() / "*.parquet");

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}
