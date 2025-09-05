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
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/search.hpp>

using namespace legate::dataframe;

template <typename T>
struct ContainsTest : public testing::Test {};

using ContainsTypes = ::testing::Types<int8_t, double, bool>;

void CompareArrow(const LogicalColumn& haystack, const LogicalColumn& needles)
{
  std::vector<arrow::Datum> args{{needles.get_arrow()}};
  auto options = arrow::compute::SetLookupOptions(
    haystack.get_arrow(), arrow::compute::SetLookupOptions::NullMatchingBehavior::EMIT_NULL);
  auto expected =
    (ARROW_RESULT(arrow::compute::CallFunction("is_in", args, &options))).make_array();
  auto result = contains(haystack, needles).get_arrow();

  EXPECT_TRUE(expected->Equals(*result))
    << "Failed contains: haystack: " << haystack.get_arrow()->ToString()
    << " needles: " << needles.get_arrow()->ToString() << " Expected: " << expected->ToString()
    << " Result: " << result->ToString();
}

TYPED_TEST_SUITE(ContainsTest, ContainsTypes);

TYPED_TEST(ContainsTest, Basic)
{
  LogicalColumn haystack(narrow<TypeParam>({1, 2, 3, 4, 5}));
  LogicalColumn needles(narrow<TypeParam>({2, 4, 6}));
  CompareArrow(haystack, needles);
}

TYPED_TEST(ContainsTest, WithNull)
{
  LogicalColumn haystack(narrow<TypeParam>({1, 2, 3, 4, 5}), {1, 0, 1, 0, 1});
  LogicalColumn needles(narrow<TypeParam>({2, 4, 6}), {1, 0, 1});
  CompareArrow(haystack, needles);
}

TYPED_TEST(ContainsTest, Scalar)
{
  LogicalColumn haystack(narrow<TypeParam>({1, 2, 3, 4, 5}));
  LogicalColumn needles(narrow<TypeParam>({2}), {}, true);
  CompareArrow(haystack, needles);
}
