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

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/sort.hpp>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;
using Table          = cudf::table;

using Order     = cudf::order;
using NullOrder = cudf::null_order;

TEST(SortTest, SimpleNoNulls)
{
  column_wrapper<int32_t> col_0{{3, 1, 2, 0, 2, 2}};
  strcol_wrapper col_1({"s1", "s1", "s1", "s0", "s0", "s0"});
  column_wrapper<int32_t> col_2{{0, 1, 2, 3, 4, 5}};

  CVector cols;
  cols.push_back(col_0.release());
  cols.push_back(col_1.release());
  cols.push_back(col_2.release());

  Table tbl(std::move(cols));

  legate::dataframe::LogicalTable lg_tbl(tbl.view(), {"a", "b", "c"});

  std::vector<Order> order({Order::ASCENDING, Order::ASCENDING, Order::ASCENDING});
  std::vector<NullOrder> null_precedence({NullOrder::AFTER, NullOrder::AFTER, NullOrder::AFTER});
  bool stable = true;

  auto expect = cudf::stable_sort(tbl, order, null_precedence);
  auto result = legate::dataframe::sort(lg_tbl, {"a", "b", "c"}, order, null_precedence, stable);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, result.get_cudf()->view());

  // Change first column to descending and use the unstable sort path:
  order           = {Order::DESCENDING, Order::ASCENDING, Order::ASCENDING};
  null_precedence = {NullOrder::AFTER, NullOrder::AFTER, NullOrder::AFTER};
  stable          = false;

  expect = cudf::sort(tbl, order, null_precedence);
  result = legate::dataframe::sort(lg_tbl, {"a", "b", "c"}, order, null_precedence, stable);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, result.get_cudf()->view());
}

TEST(SortTest, SimpleNullsSomeCols)
{
  column_wrapper<int32_t> col_0{{3, 1, 2, 0, 2, 2}, {0, 0, 1, 1, 0, 0}};
  strcol_wrapper col_1({"s1", "s1", "s1", "s0", "s0", "s0"}, {0, 0, 0, 0, 0, 1});
  column_wrapper<int32_t> col_2{{0, 1, 2, 3, 4, 5}};

  CVector cols;
  cols.push_back(col_0.release());
  cols.push_back(col_1.release());
  cols.push_back(col_2.release());

  Table tbl(std::move(cols));

  legate::dataframe::LogicalTable lg_tbl(tbl.view(), {"a", "b", "c"});

  std::vector<Order> order({Order::ASCENDING, Order::DESCENDING});
  std::vector<NullOrder> null_precedence({NullOrder::BEFORE, NullOrder::AFTER});
  bool stable = true;

  auto expect = cudf::stable_sort_by_key(tbl, tbl.select({0, 1}), order, null_precedence);
  auto result = legate::dataframe::sort(lg_tbl, {"a", "b"}, order, null_precedence, stable);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, result.get_cudf()->view());
}
