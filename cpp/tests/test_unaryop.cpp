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

#include <cstdint>
#include <legate.h>

#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/unaryop.hpp>

using namespace legate::dataframe;

template <typename T>
struct UnaryOpsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(UnaryOpsTest, cudf::test::NumericTypes);

TYPED_TEST(UnaryOpsTest, Abs)
{
  constexpr auto op = cudf::unary_operator::ABS;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({-1, -2, -3, -4});
  std::unique_ptr<cudf::column> expect = cudf::unary_operation(a, op);

  LogicalColumn res = unary_operation(LogicalColumn{a}, op);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(UnaryOpsTest, AbsWithNull)
{
  constexpr auto op = cudf::unary_operator::ABS;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({-1, -2, -3, -4}, {1, 0, 0, 1});
  std::unique_ptr<cudf::column> expect = cudf::unary_operation(a, op);

  LogicalColumn res = unary_operation(LogicalColumn{a}, op);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(UnaryOpsTest, CastFromAnyToFloat32)
{
  cudf::test::fixed_width_column_wrapper<TypeParam> a({-1, -2, -3, -4});
  std::unique_ptr<cudf::column> expect = cudf::cast(a, cudf::data_type{cudf::type_id::FLOAT32});

  LogicalColumn res = cast(LogicalColumn{a}, cudf::data_type{cudf::type_id::FLOAT32});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(UnaryOpsTest, CastFromInt16ToAny)
{
  cudf::test::fixed_width_column_wrapper<int16_t> a({-1, -2, -3, -4});
  auto to_dtype                        = cudf::data_type{cudf::type_to_id<TypeParam>()};
  std::unique_ptr<cudf::column> expect = cudf::cast(a, to_dtype);

  LogicalColumn res = cast(LogicalColumn{a}, to_dtype);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}
