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

#include <legate.h>
#include <legate/cuda/cuda.h>

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/resource_ref.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/join.hpp>
#include <legate_dataframe/utils.hpp>

using namespace legate::dataframe;

namespace {

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;

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
using CVector = std::vector<std::unique_ptr<cudf::column>>;
using Table   = cudf::table;
constexpr cudf::size_type NoneValue =
  std::numeric_limits<cudf::size_type>::min();  // TODO: how to test if this isn't public?

// This function is a wrapper around cudf's join APIs that takes the gather map
// from join APIs and materializes the table that would be created by gathering
// from the joined tables. Join APIs originally returned tables like this, but
// they were modified in https://github.com/rapidsai/cudf/pull/7454. This
// helper function allows us to avoid rewriting all our tests in terms of
// gather maps.
template <std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                    std::unique_ptr<rmm::device_uvector<cudf::size_type>>> (*join_impl)(
            cudf::table_view const& left_keys,
            cudf::table_view const& right_keys,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr),
          cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK>
std::unique_ptr<cudf::table> join_and_gather(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    join_impl(left_selected, right_selected, compare_nulls, stream, mr);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy, stream, mr);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy, stream, mr);

  return std::make_unique<cudf::table>(
    legate::dataframe::concat(left_result->release(), right_result->release()));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::inner_join>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::left_join, cudf::out_of_bounds_policy::NULLIFY>(
    left_input, right_input, left_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& full_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& full_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  return join_and_gather<cudf::full_join, cudf::out_of_bounds_policy::NULLIFY>(
    full_input, right_input, full_on, right_on, compare_nulls);
}

std::unique_ptr<cudf::table> sort_table(std::unique_ptr<cudf::table>&& tbl)
{
  return cudf::gather(tbl->view(), cudf::sorted_order(tbl->view())->view());
}
std::unique_ptr<cudf::table> sort_table(const legate::dataframe::LogicalTable& tbl)
{
  return sort_table(tbl.get_cudf());
}
std::shared_ptr<arrow::Table> arrow_join(
  const legate::dataframe::LogicalTable& lhs,
  const legate::dataframe::LogicalTable& rhs,
  const std::set<std::size_t>& left_keys,
  const std::set<std::size_t>& right_keys,
  legate::dataframe::JoinType join_type,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
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
  return ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(hashjoin)));
}
}  // namespace

auto sort_table(std::shared_ptr<arrow::Table> table)
{
  std::vector<arrow::compute::SortKey> sort_keys;
  for (const auto& key : table->ColumnNames()) {
    sort_keys.push_back(arrow::compute::SortKey{key, arrow::compute::SortOrder::Ascending});
  }
  auto indices = ARROW_RESULT(arrow::compute::SortIndices(
    table, arrow::compute::SortOptions(sort_keys, arrow::compute::NullPlacement::AtStart)));
  return ARROW_RESULT(arrow::compute::Take(table, *indices, arrow::compute::TakeOptions{})).table();
}

void test_join(const LogicalTable& a,
               const LogicalTable& b,
               std::set<std::size_t> keys_a,
               std::set<std::size_t> keys_b,
               JoinType join_type)
{
  auto result        = join(a, b, keys_a, keys_b, join_type);
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

TEST(JoinTest, LeftJoinWithNulls)
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
}

TEST(JoinTest, LeftJoinOnNulls)
{
  // Left table data - null at index 1 in column 'a'
  LogicalColumn a(std::vector<int32_t>{3, 1, 2}, {1, 0, 1});
  LogicalColumn b(std::vector<std::string>{"s0", "s1", "s2"});
  LogicalColumn c(std::vector<int32_t>{0, 1, 2});

  // Right table data - null at index 3 in column 'd'
  LogicalColumn d(std::vector<int32_t>{2, 5, 3, 7}, {1, 1, 1, 0});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s0", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 4, 2, 8});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  // Arrow join does not join on nulls so manually create expected result
  std::vector<LogicalColumn> expected_columns;
  expected_columns.emplace_back(LogicalColumn(std::vector<int32_t>{1, 2, 3}, {0, 1, 1}));
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

TEST(JoinTest, FullJoinOnNulls)
{
  // Left table data - null at index 1 in column 'a'
  LogicalColumn a(std::vector<int32_t>{3, 1}, {1, 0});
  LogicalColumn b(std::vector<std::string>{"s0", "s1"});
  LogicalColumn c(std::vector<int32_t>{0, 1});

  // Right table data - null at index 3 in column 'd'
  LogicalColumn d(std::vector<int32_t>{2, 5, 3, 7}, {1, 1, 1, 0});
  LogicalColumn e(std::vector<std::string>{"s1", "s0", "s0", "s1"});
  LogicalColumn f(std::vector<int32_t>{1, 4, 2, 8});

  LogicalTable table_0({a, b, c}, {"a", "b", "c"});
  LogicalTable table_1({d, e, f}, {"d", "e", "f"});

  auto expect =
    sort_table(full_join(table_0.get_cudf()->view(), table_1.get_cudf()->view(), {0, 1}, {0, 1}));
  auto result = sort_table(
    legate::dataframe::join(table_0, table_1, {0, 1}, {0, 1}, legate::dataframe::JoinType::FULL));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
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
  auto t0     = table_0.get_cudf();
  auto t1     = table_1.get_cudf();
  auto expect = sort_table(inner_join(*t0, *t1, {0}, {0}));

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
  auto result =
    sort_table(legate::dataframe::join(table_0,
                                       table_1,
                                       {0},
                                       {0},
                                       legate::dataframe::JoinType::INNER,
                                       /* lhs_out_columns = */ {1, 0},
                                       /* rhs_out_columns = */ std::vector<size_t>({2})));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect->select({1, 0, 5}), *result);
}

TEST(JoinTest, OutColumnNames)
{
  column_wrapper<int32_t> col0_0{{0, 1, 2}};
  column_wrapper<int32_t> col0_1{{0, 1, 2}};
  column_wrapper<int32_t> col0_2{{3, 4, 5}};

  column_wrapper<int32_t> col1_0{{2, 1, 0}};
  column_wrapper<int32_t> col1_1{{2, 1, 0}};
  column_wrapper<int32_t> col1_2{{5, 4, 3}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"key", "data0", "data1"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"key", "data0", "data1"});

  auto expect = sort_table(inner_join(t0, t1, {0}, {0}));

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
                                                   /* rhs_out_columns = */ {"data1"}));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect->select({0, 1, 5}), *result);
}

TEST(JoinTestBroacast, OutColumnNames)
{
  column_wrapper<int32_t> col0_0{{0, 1, 2}};
  column_wrapper<int32_t> col0_1{{0, 1, 2}};
  column_wrapper<int32_t> col0_2{{3, 4, 5}};

  column_wrapper<int32_t> col1_0{{2, 1, 0}};
  column_wrapper<int32_t> col1_1{{2, 1, 0}};
  column_wrapper<int32_t> col1_2{{5, 4, 3}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"key", "data0", "data1"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"key", "data0", "data1"});

  auto expect = sort_table(inner_join(t0, t1, {0}, {0}));

  std::vector<std::pair<legate::dataframe::JoinType, legate::dataframe::BroadcastInput>> bad_opts =
    {
      {legate::dataframe::JoinType::FULL, legate::dataframe::BroadcastInput::LEFT},
      {legate::dataframe::JoinType::FULL, legate::dataframe::BroadcastInput::RIGHT},
      {legate::dataframe::JoinType::LEFT, legate::dataframe::BroadcastInput::LEFT},
    };

  for (auto [how, broadcast] : bad_opts) {
    // If the names of the output columns are not unique, we expect an error
    EXPECT_THROW(legate::dataframe::join(lg_t0,
                                         lg_t1,
                                         {"key"},
                                         {"key"},
                                         how,
                                         {"data0"},
                                         {"data1"},
                                         cudf::null_equality::EQUAL,
                                         broadcast),
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
    auto result = sort_table(legate::dataframe::join(lg_t0,
                                                     lg_t1,
                                                     {"key"},
                                                     {"key"},
                                                     how,
                                                     {"data0"},
                                                     {"data1"},
                                                     cudf::null_equality::EQUAL,
                                                     broadcast));

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expect->select({0, 5}), *result);
  }
}

namespace {
std::unique_ptr<cudf::column> cudf_sequence(int64_t size)
{
  return cudf::sequence(
    size, *cudf::make_fixed_width_scalar(int64_t{0}), *cudf::make_fixed_width_scalar(int64_t{1}));
}
}  // namespace

TEST(JoinTest, InnerJoinBroadcastLHS)
{
  auto col0_0 = cudf_sequence(1);
  auto col1_0 = cudf_sequence(1e7);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(inner_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}

TEST(JoinTest, InnerJoinBroadcastRHS)
{
  auto col0_0 = cudf_sequence(1e7);
  auto col1_0 = cudf_sequence(1);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(inner_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::INNER));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}

TEST(JoinTest, LeftJoinBroadcastLHS)
{
  auto col0_0 = cudf_sequence(1);
  auto col1_0 = cudf_sequence(1e7);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(left_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::LEFT));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}

TEST(JoinTest, LeftJoinBroadcastRHS)
{
  auto col0_0 = cudf_sequence(1e7);
  auto col1_0 = cudf_sequence(1);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(left_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::LEFT));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}

TEST(JoinTest, FullJoinBroadcastLHS)
{
  auto col0_0 = cudf_sequence(1);
  auto col1_0 = cudf_sequence(1e7);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(full_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::FULL));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}

TEST(JoinTest, FullJoinBroadcastRHS)
{
  auto col0_0 = cudf_sequence(1e7);
  auto col1_0 = cudf_sequence(1);

  CVector cols0, cols1;
  cols0.push_back(std::move(col0_0));
  cols1.push_back(std::move(col1_0));

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  legate::dataframe::LogicalTable lg_t0(t0.view(), {"a"});
  legate::dataframe::LogicalTable lg_t1(t1.view(), {"b"});

  auto expect = sort_table(full_join(t0, t1, {0}, {0}));
  auto result =
    sort_table(legate::dataframe::join(lg_t0, lg_t1, {0}, {0}, legate::dataframe::JoinType::FULL));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, *result);
}
