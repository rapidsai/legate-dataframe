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

#include "gtest/gtest.h"
#include <arrow/compute/api.h>
#include <legate_dataframe/sort.hpp>

using namespace legate::dataframe;

TEST(SortTest, SimpleNoNulls)
{
  LogicalColumn a(std::vector<int>{3, 1, 2, 0, 2, 2});
  LogicalColumn b(std::vector<std::string>{"s1", "s1", "s1", "s0", "s0", "s0"});
  LogicalColumn c(std::vector<int>{0, 1, 2, 3, 4, 5});

  std::vector<std::string> column_names = {"a", "b", "c"};

  LogicalTable lg_tbl({a, b, c}, column_names);

  std::vector<bool> sort_ascending({true, true, true});
  bool nulls_at_end = false;
  bool stable       = true;
  std::vector<arrow::compute::SortKey> sort_keys;
  for (size_t i = 0; i < sort_ascending.size(); i++) {
    // Translate cudf parameters to arrow parameters
    auto order_i = sort_ascending[i] ? arrow::compute::SortOrder::Ascending
                                     : arrow::compute::SortOrder::Descending;
    sort_keys.push_back(arrow::compute::SortKey{column_names.at(i), order_i});
  }
  // Arrow does not support null_order per column, so we use the first one
  auto null_order =
    nulls_at_end ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart;

  arrow::compute::SortOptions sort_options(sort_keys, null_order);
  // auto expect = cudf::stable_sorttbl, order, null_precedence);
  auto indices = ARROW_RESULT(arrow::compute::SortIndices(lg_tbl.get_arrow(), sort_options));
  auto expect =
    ARROW_RESULT(arrow::compute::Take(lg_tbl.get_arrow(), *indices, arrow::compute::TakeOptions{}))
      .table();

  auto result =
    legate::dataframe::sort(lg_tbl, {"a", "b", "c"}, sort_ascending, nulls_at_end, stable);

  EXPECT_TRUE(result.get_arrow()->Equals(*expect))
    << "Expected: " << expect->ToString() << "\nResult: " << result.get_arrow()->ToString();
}

TEST(SortTest, SimpleNullsSomeCols)
{
  // Create LogicalColumns with nulls
  LogicalColumn a(std::vector<int>{3, 1, 2, 0, 2, 2}, std::vector<bool>{0, 0, 1, 1, 0, 0});
  LogicalColumn b(std::vector<std::string>{"s1", "s1", "s1", "s0", "s0", "s0"},
                  std::vector<bool>{0, 0, 0, 0, 0, 1});
  LogicalColumn c(std::vector<int>{0, 1, 2, 3, 4, 5});

  std::vector<std::string> column_names = {"a", "b", "c"};

  LogicalTable lg_tbl({a, b, c}, column_names);

  std::vector<bool> sort_ascending({true, false});
  bool nulls_at_end = false;
  bool stable       = true;

  // Create expected result using Arrow compute functions
  std::vector<arrow::compute::SortKey> sort_keys;
  for (size_t i = 0; i < sort_ascending.size(); i++) {
    auto order_i = sort_ascending[i] ? arrow::compute::SortOrder::Ascending
                                     : arrow::compute::SortOrder::Descending;
    sort_keys.push_back(arrow::compute::SortKey{column_names.at(i), order_i});
  }

  arrow::compute::SortOptions sort_options(
    sort_keys,
    nulls_at_end ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart);
  auto indices = ARROW_RESULT(arrow::compute::SortIndices(lg_tbl.get_arrow(), sort_options));
  auto expect =
    ARROW_RESULT(arrow::compute::Take(lg_tbl.get_arrow(), *indices, arrow::compute::TakeOptions{}))
      .table();

  auto result = legate::dataframe::sort(lg_tbl, {"a", "b"}, sort_ascending, nulls_at_end, stable);

  EXPECT_TRUE(result.get_arrow()->Equals(*expect))
    << "Expected: " << expect->ToString() << "\nResult: " << result.get_arrow()->ToString();

  // Test with null_after
  nulls_at_end = true;

  arrow::compute::SortOptions sort_options_after(
    sort_keys,
    nulls_at_end ? arrow::compute::NullPlacement::AtEnd : arrow::compute::NullPlacement::AtStart);
  auto indices_after =
    ARROW_RESULT(arrow::compute::SortIndices(lg_tbl.get_arrow(), sort_options_after));
  auto expect_after =
    ARROW_RESULT(
      arrow::compute::Take(lg_tbl.get_arrow(), *indices_after, arrow::compute::TakeOptions{}))
      .table();

  auto result_after =
    legate::dataframe::sort(lg_tbl, {"a", "b"}, sort_ascending, nulls_at_end, stable);

  EXPECT_TRUE(result_after.get_arrow()->Equals(*expect_after))
    << "Expected: " << expect_after->ToString()
    << "\nResult: " << result_after.get_arrow()->ToString();
}
