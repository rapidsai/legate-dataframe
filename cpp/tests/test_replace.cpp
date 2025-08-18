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

#include "test_utils.hpp"
#include <arrow/compute/api.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/replace.hpp>

using namespace legate::dataframe;

template <typename T>
struct NullOpsTest : public testing::Test {};

TYPED_TEST_SUITE(NullOpsTest, NumericTypes);

TYPED_TEST(NullOpsTest, FillWithScalar)
{
  auto scalar = LogicalColumn(narrow<TypeParam>({5}), {}, true);
  auto col    = LogicalColumn(narrow<TypeParam>({5, 6, 7, 8, 9}), {1, 0, 1, 0, 1});

  auto arrow_scalar = ARROW_RESULT(scalar.get_arrow()->GetScalar(0));
  auto expected =
    ARROW_RESULT(arrow::compute::CallFunction("coalesce", {col.get_arrow(), arrow_scalar}))
      .make_array();
  auto res = replace_nulls(col, scalar).get_arrow();

  EXPECT_TRUE(res->IsValid(1));
  EXPECT_TRUE(res->IsValid(3));
  EXPECT_TRUE(expected->Equals(*res));
}

TYPED_TEST(NullOpsTest, FillWithNullScalar)
{
  auto scalar = LogicalColumn(narrow<TypeParam>({5}), {0}, true);
  auto col    = LogicalColumn(narrow<TypeParam>({5, 6, 7, 8, 9}), {1, 0, 1, 0, 1});

  auto arrow_scalar = ARROW_RESULT(scalar.get_arrow()->GetScalar(0));
  auto expected =
    ARROW_RESULT(arrow::compute::CallFunction("coalesce", {col.get_arrow(), arrow_scalar}))
      .make_array();
  auto res = replace_nulls(col, scalar).get_arrow();

  EXPECT_TRUE(res->IsNull(1));
  EXPECT_TRUE(res->IsNull(3));
  EXPECT_TRUE(expected->Equals(*res));
}
