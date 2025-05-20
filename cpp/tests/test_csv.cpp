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

#include "test_utils.hpp"
#include <legate.h>

#include <legate_dataframe/csv.hpp>
#include <legate_dataframe/utils.hpp>

using namespace legate::dataframe;

template <typename T>
struct NumericCSVTest : public testing::Test {};

TYPED_TEST_SUITE(NumericCSVTest, NumericTypes);

TYPED_TEST(NumericCSVTest, ReadWrite)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  const std::vector<std::string> column_names({"a", "b"});

  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}), {1, 1, 1, 1});
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}), {1, 1, 1, 1});
  LogicalTable tbl_a({a, b}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, false);

  EXPECT_EQ(tbl_a, tbl_b);
}

TYPED_TEST(NumericCSVTest, ReadWriteSingleItem)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  // (we cannot roundtrip a single null without additional parameters).
  const std::vector<std::string> column_names({"a"});
  LogicalColumn a(narrow<TypeParam>({1}), {1});
  LogicalTable tbl_a({a}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype}, false);

  EXPECT_EQ(tbl_a, tbl_b);
}

TEST(StringsCSVTest, ReadWrite)
{
  TempDir tmp_dir;

  // NB: Columns are explicitly not null as the csv returns them as null
  LogicalColumn a({" ", "this", "is", "a", "column", "of", "strings"}, {1, 1, 1, 1, 1, 1, 1});
  const std::vector<std::string> column_names({"a"});
  LogicalTable tbl_a({a}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = tbl_a.get_column(0).cudf_type();
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype}, false);

  EXPECT_EQ(tbl_a, tbl_b);
}

TYPED_TEST(NumericCSVTest, ReadNulls)
{
  TempDir tmp_dir;
  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}), {0, 0, 1, 1});
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}), {0, 1, 1, 0});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({a, b}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, true);

  EXPECT_EQ(tbl_a, tbl_b);
}

TYPED_TEST(NumericCSVTest, ReadUseCols)
{
  TempDir tmp_dir;
  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}), {0, 0, 1, 1});
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}), {0, 1, 1, 0});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({a, b}, column_names);

  csv_write(tbl_a, tmp_dir);

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  std::vector<std::string> usecols1({"a", "b"});
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, true, ',', usecols1);

  EXPECT_EQ(tbl_a, tbl_b);

  std::vector<std::string> usecols2({"b"});
  auto tbl_c = csv_read(tmp_dir.path() / "*.csv", {dtype}, true, ',', usecols2);

  EXPECT_EQ(LogicalTable({b}, {"b"}), tbl_c);
}

TYPED_TEST(NumericCSVTest, ReadWriteWithDelimiter)
{
  TempDir tmp_dir;
  // NB: Columns are explicitly not null as the csv returns them as null
  LogicalColumn a(narrow<TypeParam>({0, 1, 2, 3}), {1, 1, 1, 1});
  LogicalColumn b(narrow<TypeParam>({4, 5, 6, 7}), {1, 1, 1, 1});
  const std::vector<std::string> column_names({"a", "b"});
  LogicalTable tbl_a({a, b}, column_names);

  csv_write(tbl_a, tmp_dir, '|');

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  auto dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto tbl_b = csv_read(tmp_dir.path() / "*.csv", {dtype, dtype}, false, '|');

  EXPECT_EQ(tbl_a, tbl_b);
}
