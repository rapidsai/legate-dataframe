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

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <cudf/binaryop.hpp>
#include <gtest/gtest.h>
#include <legate.h>

#include "test_utils.hpp"
#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

using namespace legate::dataframe;

template <typename T>
struct BinaryOpsTest : public testing::Test {};

using NumericTypesWithoutBool = ::testing::
  Types<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double>;

TYPED_TEST_SUITE(BinaryOpsTest, NumericTypesWithoutBool);

std::string op = "add";

TYPED_TEST(BinaryOpsTest, AddColCol)
{
  LogicalColumn lhs(narrow<TypeParam>({0, 1, 2, 3}));
  LogicalColumn rhs(narrow<TypeParam>({4, 5, 6, 7}));
  auto expected =
    (*arrow::compute::CallFunction(op, {lhs.get_arrow(), rhs.get_arrow()})).make_array();
  auto result = binary_operation(lhs, rhs, cudf::binary_operator::ADD, lhs.cudf_type()).get_arrow();

  EXPECT_TRUE(expected->Equals(*result));
}

TYPED_TEST(BinaryOpsTest, AddColColWithNull)
{
  LogicalColumn lhs(narrow<TypeParam>({0, 1, 2, 3}), {1, 0, 1, 0});
  LogicalColumn rhs(narrow<TypeParam>({4, 5, 6, 7}), {1, 0, 1, 0});
  auto expected =
    (*arrow::compute::CallFunction(op, {lhs.get_arrow(), rhs.get_arrow()})).make_array();
  auto result = binary_operation(lhs, rhs, cudf::binary_operator::ADD, lhs.cudf_type()).get_arrow();

  EXPECT_TRUE(expected->Equals(*result));
}

TYPED_TEST(BinaryOpsTest, AddColScalar)
{
  LogicalColumn lhs(narrow<TypeParam>({0, 1, 2, 3}));
  LogicalColumn rhs(narrow<TypeParam>({1}), {}, true);
  auto expected =
    (*arrow::compute::CallFunction(op, {lhs.get_arrow(), *rhs.get_arrow()->GetScalar(0)}))
      .make_array();
  auto result = binary_operation(lhs, rhs, cudf::binary_operator::ADD, lhs.cudf_type()).get_arrow();
  EXPECT_TRUE(expected->Equals(*result));
}

TYPED_TEST(BinaryOpsTest, AddColScalarWithNull)
{
  LogicalColumn lhs(narrow<TypeParam>({0, 1, 2, 3}), {1, 0, 1, 0});
  LogicalColumn rhs(narrow<TypeParam>({1}), {1}, true);
  auto expected =
    (*arrow::compute::CallFunction(op, {lhs.get_arrow(), *rhs.get_arrow()->GetScalar(0)}))
      .make_array();
  auto result = binary_operation(lhs, rhs, cudf::binary_operator::ADD, lhs.cudf_type()).get_arrow();
  EXPECT_TRUE(expected->Equals(*result));
}

TYPED_TEST(BinaryOpsTest, AddScalarCol)
{
  LogicalColumn lhs(narrow<TypeParam>({1}), {}, true);
  LogicalColumn rhs(narrow<TypeParam>({0, 1, 2, 3}));
  auto expected =
    (*arrow::compute::CallFunction(op, {*lhs.get_arrow()->GetScalar(0), rhs.get_arrow()}))
      .make_array();
  auto result = binary_operation(lhs, rhs, cudf::binary_operator::ADD, lhs.cudf_type()).get_arrow();
  EXPECT_TRUE(expected->Equals(*result));
}
