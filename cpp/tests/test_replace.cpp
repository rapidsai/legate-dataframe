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

#include <legate.h>

#include <cudf/replace.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/replace.hpp>

using namespace legate::dataframe;

template <typename T>
struct NullOpsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(NullOpsTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(NullOpsTest, FillWithScalar)
{
  using ScalarType = cudf::scalar_type_t<TypeParam>;
  TypeParam value  = cudf::test::make_type_param_scalar<TypeParam>(5);
  auto scalar      = ScalarType(value);
  auto lg_scalar   = LogicalColumn(scalar);

  cudf::test::fixed_width_column_wrapper<TypeParam> col({5, 6, 7, 8, 9}, {1, 0, 1, 0, 1});
  auto lg_col = LogicalColumn(col);

  auto expected = cudf::replace_nulls(col, scalar);
  auto res      = replace_nulls(lg_col, lg_scalar);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expected->view());
}

TYPED_TEST(NullOpsTest, FillWithNullScalar)
{
  using ScalarType = cudf::scalar_type_t<TypeParam>;
  TypeParam value  = cudf::test::make_type_param_scalar<TypeParam>(5);
  auto scalar      = ScalarType(value, false);  // null scalar
  auto lg_scalar   = LogicalColumn(scalar);

  cudf::test::fixed_width_column_wrapper<TypeParam> col({5, 6, 7, 8, 9}, {1, 0, 1, 0, 1});
  auto lg_col = LogicalColumn(col);

  auto expected = cudf::replace_nulls(col, scalar);
  auto res      = replace_nulls(lg_col, lg_scalar);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expected->view());
}
