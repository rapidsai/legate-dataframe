/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/binaryop.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

using namespace legate::dataframe;

template <typename T>
struct BinaryOpsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(BinaryOpsTest, cudf::test::NumericTypes);

TYPED_TEST(BinaryOpsTest, AddColCol)
{
  constexpr auto op = cudf::binary_operator::ADD;
  cudf::test::fixed_width_column_wrapper<TypeParam> lhs({0, 1, 2, 3});
  cudf::test::fixed_width_column_wrapper<TypeParam> rhs({4, 5, 6, 7});
  auto const type = static_cast<cudf::column_view>(lhs).type();

  std::unique_ptr<cudf::column> expect = cudf::binary_operation(lhs, rhs, op, type);
  LogicalColumn res = binary_operation(LogicalColumn{lhs}, LogicalColumn{rhs}, op, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(BinaryOpsTest, AddColColWithNull)
{
  constexpr auto op = cudf::binary_operator::ADD;
  cudf::test::fixed_width_column_wrapper<TypeParam> lhs({0, 1, 2, 3}, {1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<TypeParam> rhs({4, 5, 6, 7}, {1, 0, 1, 0});
  auto const type = static_cast<cudf::column_view>(lhs).type();

  std::unique_ptr<cudf::column> expect = cudf::binary_operation(lhs, rhs, op, type);
  LogicalColumn res = binary_operation(LogicalColumn{lhs}, LogicalColumn{rhs}, op, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(BinaryOpsTest, AddColScalar)
{
  constexpr auto op = cudf::binary_operator::ADD;
  cudf::test::fixed_width_column_wrapper<TypeParam> lhs({0, 1, 2, 3});
  cudf::numeric_scalar<TypeParam> rhs{1, true};
  auto const type = static_cast<cudf::column_view>(lhs).type();

  std::unique_ptr<cudf::column> expect = cudf::binary_operation(lhs, rhs, op, type);
  LogicalColumn res = binary_operation(LogicalColumn{lhs}, LogicalColumn{rhs}, op, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(BinaryOpsTest, AddColScalarWithNull)
{
  constexpr auto op = cudf::binary_operator::ADD;
  cudf::test::fixed_width_column_wrapper<TypeParam> lhs({0, 1, 2, 3}, {1, 0, 1, 0});
  cudf::numeric_scalar<TypeParam> rhs{1, true};
  auto const type = static_cast<cudf::column_view>(lhs).type();

  std::unique_ptr<cudf::column> expect = cudf::binary_operation(lhs, rhs, op, type);
  LogicalColumn res = binary_operation(LogicalColumn{lhs}, LogicalColumn{rhs}, op, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}

TYPED_TEST(BinaryOpsTest, AddScalarCol)
{
  constexpr auto op = cudf::binary_operator::ADD;
  cudf::numeric_scalar<TypeParam> lhs{1, true};
  cudf::test::fixed_width_column_wrapper<TypeParam> rhs({0, 1, 2, 3});
  auto const type = static_cast<cudf::column_view>(rhs).type();

  std::unique_ptr<cudf::column> expect = cudf::binary_operation(lhs, rhs, op, type);
  LogicalColumn res = binary_operation(LogicalColumn{lhs}, LogicalColumn{rhs}, op, type);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(res.get_cudf()->view(), expect->view());
}
