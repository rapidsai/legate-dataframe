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
#include <limits>

#include <gtest/gtest.h>
#include <legate.h>

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/filling.hpp>
#include <legate_dataframe/join.hpp>
#include <legate_dataframe/utils.hpp>

using namespace legate::dataframe;

namespace {
// Helper function to create Arrow int32 array with nulls
auto make_int32_array_with_nulls(const std::vector<int32_t>& values,
                                 const std::vector<uint8_t>& validity)
{
  arrow::Int32Builder builder;
  for (size_t i = 0; i < values.size(); ++i) {
    if (validity[i]) {
      auto status = builder.Append(values[i]);
    } else {
      auto status = builder.AppendNull();
    }
  }
  return ARROW_RESULT(builder.Finish());
}

// Helper function to create Arrow string array with nulls
auto make_string_array_with_nulls(const std::vector<std::string>& values,
                                  const std::vector<uint8_t>& validity)
{
  arrow::StringBuilder builder;
  for (size_t i = 0; i < values.size(); ++i) {
    if (validity[i]) {
      auto status = builder.Append(values[i]);
    } else {
      auto status = builder.AppendNull();
    }
  }
  return ARROW_RESULT(builder.Finish());
}

std::shared_ptr<arrow::Table> arrow_join(const legate::dataframe::LogicalTable& lhs,
                                         const legate::dataframe::LogicalTable& rhs,
                                         const std::vector<std::size_t>& left_keys,
                                         const std::vector<std::size_t>& right_keys,
                                         legate::dataframe::JoinType join_type,
                                         bool nulls_equal = true)
{
  // Convert LogicalTables to Arrow tables
  auto left_arrow  = lhs.get_arrow();
  auto right_arrow = rhs.get_arrow();

  // Set up Arrow join type mapping
  arrow::acero::JoinType arrow_join_type;
  switch (join_type) {
    case legate::dataframe::JoinType::INNER: arrow_join_type = arrow::acero::JoinType::INNER; break;
    case legate::dataframe::JoinType::LEFT:
      arrow_join_type = arrow::acero::JoinType::LEFT_OUTER;
      break;
    case legate::dataframe::JoinType::FULL:
      arrow_join_type = arrow::acero::JoinType::FULL_OUTER;
      break;
    case legate::dataframe::JoinType::SEMI:
      arrow_join_type = arrow::acero::JoinType::LEFT_SEMI;
      break;
    case legate::dataframe::JoinType::ANTI:
      arrow_join_type = arrow::acero::JoinType::LEFT_ANTI;
      break;
    default: throw std::invalid_argument("Unsupported join type");
  }

  std::vector<arrow::FieldRef> left_fields;
  std::vector<arrow::FieldRef> right_fields;

  for (const auto& key : left_keys) {
    left_fields.emplace_back(left_arrow->fields().at(key)->name());
  }
  for (const auto& key : right_keys) {
    right_fields.emplace_back(right_arrow->fields().at(key)->name());
  }

  arrow::acero::HashJoinNodeOptions join_opts{arrow_join_type, left_fields, right_fields};

  arrow::acero::Declaration left_source{"table_source",
                                        arrow::acero::TableSourceNodeOptions{left_arrow}};
  arrow::acero::Declaration right_source{"table_source",
                                         arrow::acero::TableSourceNodeOptions{right_arrow}};
  arrow::acero::Declaration hashjoin{"hashjoin", {left_source, right_source}, std::move(join_opts)};

  // Execute the join
  return ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(hashjoin), false /*use_threads*/));
}
}  // namespace

auto sort_table(std::shared_ptr<arrow::Table> table)
{
  // Arrow doesn't like it if column names are duplicated, so just rename everything then change it
  // back at the end
  auto column_names = table->ColumnNames();
  std::vector<std::string> temp_names(column_names.size());
  for (int i = 0; i < column_names.size(); i++) {
    temp_names[i] = std::to_string(i);
  }
  table = ARROW_RESULT(table->RenameColumns(temp_names));
  std::vector<arrow::compute::SortKey> sort_keys;
  for (const auto& key : temp_names) {
    sort_keys.push_back(arrow::compute::SortKey{key, arrow::compute::SortOrder::Ascending});
  }
  auto indices = ARROW_RESULT(arrow::compute::SortIndices(
    table, arrow::compute::SortOptions(sort_keys, arrow::compute::NullPlacement::AtStart)));
  auto result =
    ARROW_RESULT(arrow::compute::Take(table, *indices, arrow::compute::TakeOptions{})).table();
  return ARROW_RESULT(result->RenameColumns(column_names));
}

void test_join(const LogicalTable& a,
               const LogicalTable& b,
               std::vector<std::size_t> keys_a,
               std::vector<std::size_t> keys_b,
               JoinType join_type,
               bool nulls_equal         = true,
               BroadcastInput broadcast = BroadcastInput::AUTO)
{
  auto result        = join(a, b, keys_a, keys_b, join_type, nulls_equal, broadcast);
  auto sorted_result = sort_table(result.get_arrow());

  auto expected = arrow_join(a, b, keys_a, keys_b, join_type);

  auto sorted_expected = sort_table(expected);

  EXPECT_TRUE(sorted_result->Equals(*sorted_expected))
    << "Expected: " << sorted_expected->ToString() << "\nResult: " << sorted_result->ToString();
}

TEST(JoinTest, InnerJoinNoNulls)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 2});
  LogicalColumn b(std::vector<std::string>{"s1", "s1", "s0", "s4", "s0"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, InnerJoinOneMatch)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 3});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2", "s4", "s1"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, InnerJoinWithNulls)
{
  // Left table data
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 2});
  LogicalColumn b(std::vector<std::string>{"s1", "s1", "s0", "s4", "s0"}, {1, 1, 0, 1, 1});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});

  // Right table data
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1}, {1, 0, 1, 1, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, InnerJoinDifferentLengths)
{
  // Left table data (5 rows)
  LogicalColumn a(std::vector<int32_t>{1, 1, 0, 2, 0});
  LogicalColumn b(std::vector<std::string>{"s1", "s1", "s0", "s2", "s0"}, {1, 1, 0, 1, 1});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 3, 4});

  // Right table data (3 rows)
  LogicalColumn d(std::vector<int32_t>{1, 1, 2});
  LogicalColumn e(std::vector<std::string>{"s1", "s1", "s2"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1}, {1, 0, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, LeftJoinNoNullsWithNoCommon)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 3});
  LogicalColumn b(std::vector<std::string>{"s0", "s0", "s0", "s0", "s0"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s1", "s1", "s1", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::LEFT);
}

TEST(JoinTest, LeftJoinNoNulls)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 3});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2", "s4", "s1"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::LEFT);
}

TEST(JoinTest, LeftAntiSemiJoinWithNulls)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 2});
  LogicalColumn b(make_string_array_with_nulls({"s1", "s1", "", "s4", "s0"}, {1, 1, 0, 1, 1}));
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(make_int32_array_with_nulls({1, 0, 1, 2, 1}, {1, 0, 1, 1, 1}));

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::LEFT);
  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::ANTI);
  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::SEMI);
}

TEST(JoinTest, LeftJoinOnNulls)
{
  // Left table data - null at index 1 in column 'a'
  LogicalColumn a(std::vector<int32_t>{3, 1, 0}, {1, 0, 1});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2});

  // Right table data - null at index 3 in column 'd'
  LogicalColumn d(std::vector<int32_t>{0, 5, 3, 7}, {1, 1, 1, 0});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s0", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 4, 2, 8});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  // Arrow join does not join on nulls so manually create expected result
  std::vector<LogicalColumn> expected_columns;
  expected_columns.emplace_back(LogicalColumn(std::vector<int32_t>{1, 0, 3}, {0, 1, 1}));
  expected_columns.emplace_back(LogicalColumn(std::vector<std::string>{"s1", "s2", "s0"}));
  expected_columns.emplace_back(LogicalColumn(std::vector<int32_t>{1, 2, 0}));
  expected_columns.emplace_back(LogicalColumn(std::vector<int32_t>{8, 1, 2}, {1, 0, 1}));
  LogicalTable expected(expected_columns, std::vector<std::string>{"a", "b", "c", "f"});
  auto result = legate::dataframe::join(table_0,
                                        table_1,
                                        {"a", "b"},
                                        {"d", "e"},
                                        legate::dataframe::JoinType::LEFT,
                                        {"a", "b", "c"},
                                        {"f"});

  auto sorted_expected = sort_table(expected.get_arrow());
  auto sorted_result   = sort_table(result.get_arrow());

  EXPECT_TRUE(sorted_result->Equals(*sorted_expected))
    << "Expected: " << sorted_expected->ToString() << "\nResult: " << sorted_result->ToString();
}

TEST(JoinTest, FullJoinNoNulls)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 3});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2", "s4", "s1"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(std::vector<int32_t>{2, 2, 0, 4, 3});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::FULL);
}

TEST(JoinTest, FullJoinWithNulls)
{
  LogicalColumn a(std::vector<int32_t>{3, 1, 2, 0, 3});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2", "s4", "s1"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2, 4, 1});
  LogicalColumn d(make_int32_array_with_nulls({2, 2, 0, 4, 3}, {1, 1, 1, 0, 1}));
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s1", "s2", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 0, 1, 2, 1});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  test_join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::FULL);
}

TEST(JoinTest, OutColumnIndices)
{
  LogicalColumn a(std::vector<int32_t>{0, 1, 2});
  LogicalColumn b(std::vector<int32_t>{0, 1, 2});
  LogicalColumn c(std::vector<int32_t>{3, 4, 5});
  LogicalColumn d(std::vector<int32_t>{2, 1, 0});
  LogicalColumn e(std::vector<int32_t>{2, 1, 0});
  LogicalColumn f(std::vector<int32_t>{5, 4, 3});

  LogicalTable table_0({a, b, c}, {"key", "data0", "data1"});
  LogicalTable table_1({d, e, f}, {"key", "data0", "data1"});

  // Get cudf tables for comparison with expected result
  auto expect =
    sort_table(arrow_join(table_0, table_1, {0}, {0}, legate::dataframe::JoinType::INNER));

  // If the names of the output columns are not unique, we expect an error
  EXPECT_THROW(legate::dataframe::join(table_0,
                                       table_1,
                                       {0},
                                       {0},
                                       legate::dataframe::JoinType::INNER,
                                       /* lhs_out_columns = */ {0, 1},
                                       /* rhs_out_columns = */ {2, 1}),
               std::invalid_argument);

  // By specifying the output columns, we can join tables that would otherwise have
  // name conflicts
  auto result = sort_table(legate::dataframe::join(table_0,
                                                   table_1,
                                                   {0},
                                                   {0},
                                                   legate::dataframe::JoinType::INNER,
                                                   /* lhs_out_columns = */ {1, 0},
                                                   /* rhs_out_columns = */ std::vector<size_t>({2}))
                             .get_arrow());

  EXPECT_TRUE(ARROW_RESULT(expect->SelectColumns({1, 0, 5}))->Equals(*result));
}

TEST(JoinTest, OutColumnNames)
{
  LogicalColumn col0_0(std::vector<int32_t>{0, 1, 2});
  LogicalColumn col0_1(std::vector<int32_t>{0, 1, 2});
  LogicalColumn col0_2(std::vector<int32_t>{3, 4, 5});

  LogicalColumn col1_0(std::vector<int32_t>{2, 1, 0});
  LogicalColumn col1_1(std::vector<int32_t>{2, 1, 0});
  LogicalColumn col1_2(std::vector<int32_t>{5, 4, 3});

  legate::dataframe::LogicalTable lg_t0({col0_0, col0_1, col0_2}, {"key", "data0", "data1"});
  legate::dataframe::LogicalTable lg_t1({col1_0, col1_1, col1_2}, {"key", "data0", "data1"});

  auto expect = sort_table(arrow_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER));

  // If the names of the output columns are not unique, we expect an error
  EXPECT_THROW(legate::dataframe::join(lg_t0,
                                       lg_t1,
                                       {"key"},
                                       {"key"},
                                       legate::dataframe::JoinType::INNER,
                                       /* lhs_out_columns = */ {"data0", "key"},
                                       /* rhs_out_columns = */ {"data1", "data0"}),
               std::invalid_argument);

  // By specifying the output columns, we can join tables that would otherwise have
  // name conflicts
  auto result = sort_table(legate::dataframe::join(lg_t0,
                                                   lg_t1,
                                                   {"key", "data0"},
                                                   {"key", "data0"},
                                                   legate::dataframe::JoinType::INNER,
                                                   /* lhs_out_columns = */ {"key", "data0"},
                                                   /* rhs_out_columns = */ {"data1"})
                             .get_arrow());

  EXPECT_TRUE(ARROW_RESULT(expect->SelectColumns({0, 1, 5}))->Equals(*result));
}

TEST(JoinTestBroacast, OutColumnNames)
{
  LogicalColumn col0_0{std::vector<int>{0, 1, 2}};
  LogicalColumn col0_1{std::vector<int>{0, 1, 2}};
  LogicalColumn col0_2{std::vector<int>{3, 4, 5}};

  LogicalColumn col1_0{std::vector<int>{2, 1, 0}};
  LogicalColumn col1_1{std::vector<int>{2, 1, 0}};
  LogicalColumn col1_2{std::vector<int>{5, 4, 3}};

  legate::dataframe::LogicalTable lg_t0({col0_0, col0_1, col0_2}, {"key", "data0", "data1"});
  legate::dataframe::LogicalTable lg_t1({col1_0, col1_1, col1_2}, {"key", "data0", "data1"});

  auto expect = sort_table(arrow_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER));

  std::vector<std::pair<legate::dataframe::JoinType, legate::dataframe::BroadcastInput>> bad_opts =
    {
      {legate::dataframe::JoinType::FULL, legate::dataframe::BroadcastInput::LEFT},
      {legate::dataframe::JoinType::FULL, legate::dataframe::BroadcastInput::RIGHT},
      {legate::dataframe::JoinType::LEFT, legate::dataframe::BroadcastInput::LEFT},
    };

  for (auto [how, broadcast] : bad_opts) {
    // If the names of the output columns are not unique, we expect an error
    EXPECT_THROW(legate::dataframe::join(
                   lg_t0, lg_t1, {"key"}, {"key"}, how, {"data0"}, {"data1"}, true, broadcast),
                 std::runtime_error);
  }

  std::vector<std::pair<legate::dataframe::JoinType, legate::dataframe::BroadcastInput>> good_opts =
    {
      {legate::dataframe::JoinType::FULL, legate::dataframe::BroadcastInput::AUTO},
      {legate::dataframe::JoinType::INNER, legate::dataframe::BroadcastInput::AUTO},
      {legate::dataframe::JoinType::LEFT, legate::dataframe::BroadcastInput::AUTO},
      {legate::dataframe::JoinType::INNER, legate::dataframe::BroadcastInput::LEFT},
      {legate::dataframe::JoinType::INNER, legate::dataframe::BroadcastInput::RIGHT},
      {legate::dataframe::JoinType::LEFT, legate::dataframe::BroadcastInput::RIGHT},
    };

  for (auto [how, broadcast] : good_opts) {
    auto result =
      sort_table(legate::dataframe::join(
                   lg_t0, lg_t1, {"key"}, {"key"}, how, {"data0"}, {"data1"}, true, broadcast)
                   .get_arrow());

    EXPECT_TRUE(ARROW_RESULT(expect->SelectColumns({1, 5}))->Equals(*result));
  }
}

TEST(JoinTest, InnerJoinBroadcastLHS)
{
  auto col0_0 = legate::dataframe::sequence(1, 0);
  auto col1_0 = legate::dataframe::sequence(1e7, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});

  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, InnerJoinBroadcastRHS)
{
  auto col0_0 = legate::dataframe::sequence(1e7, 0);
  auto col1_0 = legate::dataframe::sequence(1, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});

  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER);
}

TEST(JoinTest, LeftJoinBroadcastLHS)
{
  auto col0_0 = legate::dataframe::sequence(1, 0);
  auto col1_0 = legate::dataframe::sequence(1e7, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});

  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::LEFT);
}

TEST(JoinTest, LeftJoinBroadcastRHS)
{
  auto col0_0 = legate::dataframe::sequence(1e7, 0);
  auto col1_0 = legate::dataframe::sequence(1, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});

  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::LEFT);
}

TEST(JoinTest, FullJoinBroadcastLHS)
{
  auto col0_0 = legate::dataframe::sequence(1, 0);
  auto col1_0 = legate::dataframe::sequence(1e7, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});
  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::FULL);
}

TEST(JoinTest, FullJoinBroadcastRHS)
{
  auto col0_0 = legate::dataframe::sequence(1e7, 0);
  auto col1_0 = legate::dataframe::sequence(1, 0);

  legate::dataframe::LogicalTable lg_t0({col0_0}, {"a"});
  legate::dataframe::LogicalTable lg_t1({col1_0}, {"b"});

  test_join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::FULL);
}
