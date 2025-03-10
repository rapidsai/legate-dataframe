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

#include <cstdint>
#include <iostream>

#include <legate.h>

#include <legate_dataframe/csv.hpp>
#include <legate_dataframe/utils.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

using namespace legate::dataframe;

template <typename T>
struct NumericCSVTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NumericCSVTest, cudf::test::NumericTypes);

TYPED_TEST(NumericCSVTest, ReadWrite)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3}, {1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7}, {1, 1, 1, 1});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, false);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TYPED_TEST(NumericCSVTest, ReadWriteSingleItem)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  // (we cannot roundtrip a single null without additional parameters).
  cudf::test::fixed_width_column_wrapper<TypeParam> a({1}, {1});
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({LogicalColumn{a}}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype}, false);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TEST(StringsCSVTest, ReadWrite)
{
  TempDir tmp_dir;

  // NB: Columns are explicitly not null as the csv returns them as null
  cudf::test::strings_column_wrapper s({" ", "this", "is", "a", "column", "of", "strings"},
                                       {1, 1, 1, 1, 1, 1, 1});
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({LogicalColumn{s}}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = tbl_a.get_column(0).cudf_type();
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype}, false);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TYPED_TEST(NumericCSVTest, ReadNulls)
{
  TempDir tmp_dir;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3}, {0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7}, {0, 1, 1, 0});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, true);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}

TYPED_TEST(NumericCSVTest, ReadUseCols)
{
  TempDir tmp_dir;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3}, {0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7}, {0, 1, 1, 0});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  std::vector<std::string> usecols1({"a", "b"});
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, true, ',', usecols1);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());

  std::vector<std::string> usecols2({"b"});
  auto tbl_c = csv_read(tmp_dir.path() / "*.csv", {dtype}, true, ',', usecols2);

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_c.get_cudf()->view(), tbl_a.select({"b"}).get_cudf()->view());
  EXPECT_TRUE(tbl_c.get_column_name_vector() == usecols2);
}

TYPED_TEST(NumericCSVTest, ReadWriteWithDelimiter)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3}, {1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({4, 5, 6, 7}, {1, 1, 1, 1});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({LogicalColumn{a}, LogicalColumn{b}}, column_names);

  csv_write(tbl_a, tmp_dir, '|');

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, false, '|');

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl_a.get_cudf()->view(), tbl_b.get_cudf()->view());
  EXPECT_TRUE(tbl_a.get_column_names() == tbl_b.get_column_names());
}
