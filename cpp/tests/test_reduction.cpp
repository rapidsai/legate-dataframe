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

#include "test_utils.hpp"
#include <arrow/compute/api.h>
#include <legate.h>

#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/reduction.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/reduction.hpp>

using namespace legate::dataframe;

template <typename T>
struct ReductionTest : public testing::Test {};

TYPED_TEST_SUITE(ReductionTest, NumericTypes);

TYPED_TEST(ReductionTest, Max)
{
  auto agg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  LogicalColumn col(narrow<TypeParam>({5, 6, 7, 8, 9}), {1, 0, 1, 0, 1});

  auto res_dtype = cudf::detail::target_type(col.cudf_type(), agg->kind);
  auto res       = reduce(col, "max", res_dtype);

  auto expected = ARROW_RESULT(arrow::compute::CallFunction("max", {col.get_arrow()})).scalar();
  auto expected_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*expected, 1));

  EXPECT_TRUE(res.get_arrow()->Equals(expected_array))
    << "Expected: " << expected_array->ToString() << ", got: " << res.get_arrow()->ToString();
}

TYPED_TEST(ReductionTest, Mean)
{
  LogicalColumn col(narrow<TypeParam>({5, 6, 7, 8, 9}), {1, 0, 1, 0, 1});
  auto agg            = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
  auto res_dtype      = cudf::detail::target_type(col.cudf_type(), agg->kind);
  auto res            = reduce(col, "mean", res_dtype);
  auto expected       = ARROW_RESULT(arrow::compute::Mean(col.get_arrow())).scalar();
  auto expected_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*expected, 1));

  EXPECT_TRUE(res.get_arrow()->ApproxEquals(expected_array))
    << "Expected: " << expected_array->ToString() << ", got: " << res.get_arrow()->ToString();
}

TYPED_TEST(ReductionTest, EmptyMax)
{
  LogicalColumn col(std::vector<TypeParam>{}, std::vector<bool>{});
  auto agg       = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(col.cudf_type(), agg->kind);
  auto res       = reduce(col, "max", res_dtype);
  auto expected  = ARROW_RESULT(arrow::compute::CallFunction("max", {col.get_arrow()})).scalar();
  auto expected_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*expected, 1));
  EXPECT_TRUE(res.get_arrow()->Equals(expected_array))
    << "Expected: " << expected_array->ToString() << ", got: " << res.get_arrow()->ToString();
}

TYPED_TEST(ReductionTest, AllNullSum)
{
  LogicalColumn col(narrow<TypeParam>({1, 2, 3, 4}), {1, 1, 1, 1});
  auto agg       = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto res_dtype = cudf::detail::target_type(col.cudf_type(), agg->kind);
  auto res       = reduce(col, "sum", res_dtype);
  auto expected  = ARROW_RESULT(arrow::compute::CallFunction("sum", {col.get_arrow()})).scalar();
  auto expected_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*expected, 1));
  // Cast expected type
  expected_array =
    ARROW_RESULT(arrow::compute::Cast(expected_array, res.get_arrow()->type())).make_array();

  EXPECT_TRUE(res.get_arrow()->Equals(expected_array))
    << "Expected: " << expected_array->ToString() << ", got: " << res.get_arrow()->ToString();
}
