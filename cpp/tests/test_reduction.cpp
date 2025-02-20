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

#include <legate.h>

#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/reduction.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/reduction.hpp>

using namespace legate::dataframe;

template <typename T>
struct ReductionTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ReductionTest, cudf::test::NumericTypes);

TYPED_TEST(ReductionTest, Max)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({5, 6, 7, 8, 9}, {1, 0, 1, 0, 1});
  auto const type = static_cast<cudf::column_view>(col).type();
  auto lg_col     = LogicalColumn{col};

  auto agg       = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(type, agg->kind);

  auto expected   = cudf::reduce(col, *agg, res_dtype);
  auto res        = reduce(lg_col, *agg, res_dtype);
  auto res_scalar = res.get_cudf_scalar();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::make_column_from_scalar(*res_scalar, 1)->view(),
                                 cudf::make_column_from_scalar(*expected, 1)->view());
}

TYPED_TEST(ReductionTest, Mean)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({5, 6, 7, 8, 9}, {1, 0, 1, 0, 1});
  auto const type = static_cast<cudf::column_view>(col).type();
  auto lg_col     = LogicalColumn{col};

  auto agg       = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(type, agg->kind);

  auto expected   = cudf::reduce(col, *agg, res_dtype);
  auto res        = reduce(lg_col, *agg, res_dtype);
  auto res_scalar = res.get_cudf_scalar();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::make_column_from_scalar(*res_scalar, 1)->view(),
                                 cudf::make_column_from_scalar(*expected, 1)->view());
}

TYPED_TEST(ReductionTest, EmptyMax)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({});
  auto const type = static_cast<cudf::column_view>(col).type();
  auto lg_col     = LogicalColumn{col};

  auto agg       = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(type, agg->kind);

  auto expected   = cudf::reduce(col, *agg, res_dtype);
  auto res        = reduce(lg_col, *agg, res_dtype);
  auto res_scalar = res.get_cudf_scalar();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::make_column_from_scalar(*res_scalar, 1)->view(),
                                 cudf::make_column_from_scalar(*expected, 1)->view());
}

TYPED_TEST(ReductionTest, AllNullSum)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> col({1, 2, 3, 4}, {1, 1, 1, 1});
  auto const type = static_cast<cudf::column_view>(col).type();
  auto lg_col     = LogicalColumn{col};

  auto agg       = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(type, agg->kind);

  auto expected   = cudf::reduce(col, *agg, res_dtype);
  auto res        = reduce(lg_col, *agg, res_dtype);
  auto res_scalar = res.get_cudf_scalar();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::make_column_from_scalar(*res_scalar, 1)->view(),
                                 cudf::make_column_from_scalar(*expected, 1)->view());
}
