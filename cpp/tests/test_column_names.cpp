/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <glob.h>

#include <legate.h>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/filling.hpp>
#include <legate_dataframe/join.hpp>

using namespace legate::dataframe;

// Generate unique columns
std::vector<LogicalColumn> gen_columns(size_t num_columns)
{
  std::vector<LogicalColumn> ret;
  for (size_t i = 0; i < num_columns; ++i) {
    ret.emplace_back(sequence(100, 0));
  }
  return ret;
}

TEST(ColumnNameTest, FromColumnVector)
{
  const std::vector<std::string> names = {"a", "b", "c"};
  std::vector<LogicalColumn> cols      = gen_columns(names.size());
  auto tbl                             = LogicalTable(cols, names);

  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_TRUE(&tbl.get_column(names[i]) == &tbl.get_column(i));
  }

  auto tbl_like = LogicalTable::empty_like(tbl);
  EXPECT_TRUE(tbl_like.get_column_names() == tbl.get_column_names());
}

TEST(ColumnNameTest, FromCudf)
{
  const std::vector<std::string> names = {"a", "b"};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<int32_t> a({0, 1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> b({5, 6, 7, 8, 9});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table original(std::move(cols));

  auto tbl = LogicalTable(original, names);
  for (size_t i = 0; i < names.size(); ++i) {
    const auto& col = tbl.get_column(names[i]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(col.get_cudf()->view(), original.get_column(i));
  }

  auto tbl_like = LogicalTable::empty_like(original.view(), names);
  EXPECT_TRUE(tbl_like.get_column_names() == tbl.get_column_names());
}

TEST(ColumnNameTest, Join)
{
  const std::vector<std::string> lhs_names = {"a", "b"};
  const std::vector<std::string> rhs_names = {"c", "d"};

  auto lhs = LogicalTable(gen_columns(lhs_names.size()), lhs_names);
  auto rhs = LogicalTable(gen_columns(rhs_names.size()), rhs_names);
  auto res = join(lhs, rhs, {0}, {1}, legate::dataframe::JoinType::INNER);

  const std::map<std::string, size_t> expect = {{"a", 0}, {"b", 1}, {"c", 2}, {"d", 3}};
  EXPECT_TRUE(res.get_column_names() == expect);
}
