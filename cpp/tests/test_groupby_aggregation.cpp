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

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>

#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <legate.h>
#include <legate_dataframe/sort.hpp>

#include <legate_dataframe/groupby_aggregation.hpp>

using namespace legate::dataframe;

template <typename V>
struct GroupByAggregationTest : public testing::Test {};

using K     = int32_t;
using Types = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(GroupByAggregationTest, Types);

namespace {

auto sort_table(std::shared_ptr<arrow::Table> table, const std::vector<std::string>& keys)
{
  std::vector<arrow::compute::SortKey> sort_keys;
  for (const auto& key : keys) {
    sort_keys.push_back(arrow::compute::SortKey{key, arrow::compute::SortOrder::Ascending});
  }
  // Arrow does not support null_order per column, so we use the first one
  auto indices = ARROW_RESULT(arrow::compute::SortIndices(
    table, arrow::compute::SortOptions(sort_keys, arrow::compute::NullPlacement::AtStart)));
  return ARROW_RESULT(arrow::compute::Take(table, *indices, arrow::compute::TakeOptions{})).table();
}

void assert_arrow_tables_equal(const std::vector<std::string>& keys,
                               std::shared_ptr<arrow::Table> expected,
                               std::shared_ptr<arrow::Table> actual)
{
  // Sort based on keys
  auto expected_sorted = sort_table(expected, keys);
  auto actual_sorted   = sort_table(actual, keys);

  // Compare each column
  for (auto name : expected_sorted->ColumnNames()) {
    auto expected_col = expected_sorted->GetColumnByName(name);
    auto actual_col   = actual_sorted->GetColumnByName(name);

    // Cast expected to same type if needed
    if (expected_col->type() != actual_col->type()) {
      auto cast = ARROW_RESULT(arrow::compute::Cast(*arrow::Concatenate(expected_col->chunks()),
                                                    actual_col->type()))
                    .make_array();
      expected_col = std::make_shared<arrow::ChunkedArray>(cast);
    }

    EXPECT_TRUE(expected_col->ApproxEquals(*actual_col));
  }
}

}  // namespace

TYPED_TEST(GroupByAggregationTest, single_sum_with_nulls)
{
  using V = TypeParam;

  auto keys_column = LogicalColumn(narrow<K>({1, 2, 3, 1, 2, 2, 1, 3, 3, 2}));
  auto values_column =
    LogicalColumn(narrow<V>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), {0, 1, 0, 1, 1, 1, 0, 1, 0, 0});
  const std::vector<std::string> names({"key", "value"});
  auto table = LogicalTable({keys_column, values_column}, names);

  arrow::compute::Aggregate aggregate("hash_sum", {"value"}, "sum");
  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(table.get_arrow())},
     {"aggregate", arrow::acero::AggregateNodeOptions({aggregate}, {"key"})}});
  auto expected =
    ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan), false /*use_threads*/));

  auto result = groupby_aggregation(table, {"key"}, {std::make_tuple("value", "sum", "sum")});

  result = legate::dataframe::sort(result, {"key"}, {true}, true, true);

  auto result_arrow = result.get_arrow();

  assert_arrow_tables_equal({"key"}, expected, result_arrow);
}

TYPED_TEST(GroupByAggregationTest, nunique_and_max)
{
  using V = TypeParam;

  auto keys_column  = LogicalColumn(narrow<K>({1, 2, 3, 1, 2, 2, 1, 3, 3, 2}));
  auto vals1_column = LogicalColumn(narrow<V>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  auto vals2_column = LogicalColumn(narrow<V>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));

  const std::vector<std::string> names({"key", "vals1", "vals2"});
  auto table = LogicalTable({keys_column, vals1_column, vals2_column}, names);

  // Create expected result using Arrow
  arrow::compute::Aggregate nunique_agg1("hash_count_distinct", {"vals1"}, "nunique1");
  arrow::compute::Aggregate max_agg1("hash_max", {"vals1"}, "max1");
  arrow::compute::Aggregate nunique_agg2("hash_count_distinct", {"vals2"}, "nunique2");
  arrow::compute::Aggregate max_agg2("hash_max", {"vals2"}, "max2");

  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(table.get_arrow())},
     {"aggregate",
      arrow::acero::AggregateNodeOptions({nunique_agg1, max_agg1, nunique_agg2, max_agg2},
                                         {"key"})}});
  auto expected =
    ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan), false /*use_threads*/));

  auto result = groupby_aggregation(table,
                                    {"key"},
                                    {std::make_tuple("vals1", "count_distinct", "nunique1"),
                                     std::make_tuple("vals1", "max", "max1"),
                                     std::make_tuple("vals2", "count_distinct", "nunique2"),
                                     std::make_tuple("vals2", "max", "max2")});

  result = legate::dataframe::sort(result, {"key"}, {true}, true, true);

  assert_arrow_tables_equal({"key"}, expected, result.get_arrow());
}

TYPED_TEST(GroupByAggregationTest, stddev_and_mean_with_multiple_keys)
{
  using V           = TypeParam;
  auto keys1_column = LogicalColumn(narrow<K>({1, 2, 3, 1, 2, 1, 1, 3, 1, 2}));
  auto vals1_column = LogicalColumn(narrow<V>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  auto keys2_column = LogicalColumn(narrow<K>({1, 2, 3, 1, 1, 2, 1, 3, 2, 2}));
  auto vals2_column = LogicalColumn(narrow<V>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));

  const std::vector<std::string> names({"keys1", "vals1", "keys2", "vals2"});
  auto table = LogicalTable({keys1_column, vals1_column, keys2_column, vals2_column}, names);

  // Create expected result using Arrow
  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(table.get_arrow())},
     {"aggregate",
      arrow::acero::AggregateNodeOptions({{"hash_stddev", "vals1", "stddev1"},
                                          {"hash_mean", "vals1", "mean1"},
                                          {"hash_stddev", "vals2", "stddev2"},
                                          {"hash_mean", "vals2", "mean2"}},
                                         {"keys1", "keys2"})}});
  auto expected =
    ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan), false /*use_threads*/));

  auto result = groupby_aggregation(table,
                                    {"keys1", "keys2"},
                                    {std::make_tuple("vals1", "stddev", "stddev1"),
                                     std::make_tuple("vals1", "mean", "mean1"),
                                     std::make_tuple("vals2", "stddev", "stddev2"),
                                     std::make_tuple("vals2", "mean", "mean2")});

  result = legate::dataframe::sort(result, {"keys1", "keys2"}, {true, true}, true, true);

  assert_arrow_tables_equal({"keys1", "keys2"}, expected, result.get_arrow());
}
