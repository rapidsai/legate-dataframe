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

#include "gmock/gmock-matchers.h"
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <legate.h>

#include "test_utils.hpp"
#include <legate_dataframe/copying.hpp>
#include <legate_dataframe/core/column.hpp>

using namespace legate::dataframe;

template <typename T>
struct CopyingTest : public testing::Test {};

using CopyingTypes = ::testing::Types<int8_t, double, bool>;

void CompareArrow(const LogicalColumn& cond, const LogicalColumn& lhs, const LogicalColumn& rhs)
{
  std::vector<arrow::Datum> args;
  args.reserve(3);
  for (auto& column : {&cond, &lhs, &rhs}) {
    if (column->num_rows() == 1) {
      auto scalar = ARROW_RESULT(column->get_arrow()->GetScalar(0));
      args.emplace_back(scalar);
    } else {
      args.emplace_back(column->get_arrow());
    }
  }

  auto expected = (ARROW_RESULT(arrow::compute::CallFunction("if_else", args))).make_array();
  auto result   = copy_if_else(cond, lhs, rhs).get_arrow();

  EXPECT_TRUE(expected->Equals(*result))
    << "Failed copy_if_else: " << cond.get_arrow()->ToString()
    << " LHS: " << lhs.get_arrow()->ToString() << " RHS: " << rhs.get_arrow()->ToString()
    << " Expected: " << expected->ToString() << " Result: " << result->ToString();
}

TYPED_TEST_SUITE(CopyingTest, CopyingTypes);

TYPED_TEST(CopyingTest, ColCol)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}));
  LogicalColumn rhs(narrow<TypeParam>({5, 6, 7, 8}));
  LogicalColumn cond(narrow<bool>({true, false, true, false}));
  CompareArrow(cond, lhs, rhs);
}

TYPED_TEST(CopyingTest, ColColWithNull)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}), {1, 0, 1, 0});
  LogicalColumn rhs(narrow<TypeParam>({5, 6, 7, 8}), {1, 0, 1, 0});
  LogicalColumn cond(narrow<bool>({true, true, false, false}));
  CompareArrow(cond, lhs, rhs);
}

TYPED_TEST(CopyingTest, ColScalar)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}));
  LogicalColumn rhs(narrow<TypeParam>({1}), {}, true);
  LogicalColumn cond(narrow<bool>({true, true, false, false}));
  CompareArrow(cond, lhs, rhs);
}

TYPED_TEST(CopyingTest, ScalarCol)
{
  LogicalColumn lhs(narrow<TypeParam>({1}), {1}, true);
  LogicalColumn rhs(narrow<TypeParam>({1, 2, 3, 4}));
  LogicalColumn cond(narrow<bool>({true, true, false, false}));
  CompareArrow(cond, lhs, rhs);
}

TYPED_TEST(CopyingTest, ScalarScalar)
{
  LogicalColumn lhs(narrow<TypeParam>({1}), {}, true);
  LogicalColumn rhs(narrow<TypeParam>({2}), {}, true);
  LogicalColumn cond(narrow<bool>({true, true, false, false}));
  CompareArrow(cond, lhs, rhs);
}
