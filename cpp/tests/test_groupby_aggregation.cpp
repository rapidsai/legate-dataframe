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

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/groupby_aggregation.hpp>

using namespace legate::dataframe;

template <typename V>
struct GroupByAggregationTest : public cudf::test::BaseFixture {};

using K      = int32_t;
using dtypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(GroupByAggregationTest, dtypes);

namespace {

std::unique_ptr<cudf::table> sort_table(const cudf::table& tbl)
{
  return cudf::gather(tbl.view(), cudf::sorted_order(tbl.view())->view());
}

std::unique_ptr<cudf::table> sort_result(const LogicalTable& lg_table)
{
  return sort_table(cudf::table(lg_table.get_cudf()->release()));
}

std::unique_ptr<cudf::table> sort_result(
  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>>&& res)
{
  auto unique_keys = res.first->release();
  auto values      = std::move(res.second.at(0).results);
  for (size_t i = 1; i < res.second.size(); ++i) {
    values = concat(std::move(values), std::move(res.second.at(i).results));
  }
  auto keys_and_values = concat(std::move(unique_keys), std::move(values));
  return sort_table(cudf::table(std::move(keys_and_values)));
}

}  // namespace

TYPED_TEST(GroupByAggregationTest, single_sum_with_nulls)
{
  using V  = TypeParam;
  auto SUM = cudf::aggregation::Kind::SUM;

  cudf::test::fixed_width_column_wrapper<K> _keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> _vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                  {0, 1, 0, 1, 1, 1, 0, 1, 0, 0});
  auto keys = cudf::table_view({_keys});
  auto vals = cudf::column_view(_vals);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values       = vals;
  requests[0].aggregations = make_groupby_aggregations({SUM});

  cudf::groupby::groupby gb_obj(keys);
  auto expect = sort_result(gb_obj.aggregate(requests));

  std::vector<LogicalColumn> lg_columns    = {keys.column(0), vals};
  std::vector<std::string> lg_column_names = {"key", "val"};
  LogicalTable lg_table(std::move(lg_columns), std::move(lg_column_names));

  auto result = groupby_aggregation(lg_table, {"key"}, {std::make_tuple("val", SUM, "sum")});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect->view(), sort_result(result)->view());
}

TYPED_TEST(GroupByAggregationTest, nunique_and_max)
{
  using V      = TypeParam;
  auto NUNIQUE = cudf::aggregation::Kind::NUNIQUE;
  auto MAX     = cudf::aggregation::Kind::MAX;

  cudf::test::fixed_width_column_wrapper<K> _keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> _vals1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> _vals2{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  auto keys  = cudf::table_view({_keys});
  auto vals1 = cudf::column_view(_vals1);
  auto vals2 = cudf::column_view(_vals2);

  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values       = vals1;
  requests[0].aggregations = make_groupby_aggregations({NUNIQUE, MAX});
  requests[1].values       = vals2;
  requests[1].aggregations = make_groupby_aggregations({NUNIQUE, MAX});

  cudf::groupby::groupby gb_obj(keys);
  auto expect = sort_result(gb_obj.aggregate(requests));

  std::vector<LogicalColumn> lg_columns    = {keys.column(0), vals1, vals2};
  std::vector<std::string> lg_column_names = {"key", "vals1", "vals2"};
  LogicalTable lg_table(std::move(lg_columns), std::move(lg_column_names));

  auto result = groupby_aggregation(lg_table,
                                    {"key"},
                                    {std::make_tuple("vals1", NUNIQUE, "nunique1"),
                                     std::make_tuple("vals1", MAX, "max1"),
                                     std::make_tuple("vals2", NUNIQUE, "nunique2"),
                                     std::make_tuple("vals2", MAX, "max2")});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect->view(), sort_result(result)->view());
}

TYPED_TEST(GroupByAggregationTest, median_and_mean_with_multiple_keys)
{
  using V     = TypeParam;
  auto MEDIAN = cudf::aggregation::Kind::MEDIAN;
  auto MEAN   = cudf::aggregation::Kind::MEAN;

  cudf::test::fixed_width_column_wrapper<K> _keys1{1, 2, 3, 1, 2, 1, 1, 3, 1, 2};
  cudf::test::fixed_width_column_wrapper<K> _keys2{1, 2, 3, 1, 1, 2, 1, 3, 2, 2};
  cudf::test::fixed_width_column_wrapper<V> _vals1{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<V> _vals2{10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  auto keys  = cudf::table_view({_keys1, _keys2});
  auto vals1 = cudf::column_view(_vals1);
  auto vals2 = cudf::column_view(_vals2);

  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values       = vals1;
  requests[0].aggregations = make_groupby_aggregations({MEDIAN, MEAN});
  requests[1].values       = vals2;
  requests[1].aggregations = make_groupby_aggregations({MEDIAN, MEAN});

  cudf::groupby::groupby gb_obj(keys);
  std::unique_ptr<cudf::table> expect = sort_result(gb_obj.aggregate(requests));

  std::vector<LogicalColumn> lg_columns    = {keys.column(0), vals1, keys.column(1), vals2};
  std::vector<std::string> lg_column_names = {"keys1", "vals1", "keys2", "vals2"};
  LogicalTable lg_table(std::move(lg_columns), std::move(lg_column_names));

  auto result = groupby_aggregation(lg_table,
                                    {"keys1", "keys2"},
                                    {std::make_tuple("vals1", MEDIAN, "nunique1"),
                                     std::make_tuple("vals1", MEAN, "max1"),
                                     std::make_tuple("vals2", MEDIAN, "nunique2"),
                                     std::make_tuple("vals2", MEAN, "max2")});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expect->view(), sort_result(result)->view());
}
