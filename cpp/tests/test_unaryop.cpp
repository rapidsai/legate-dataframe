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

#include "test_utils.hpp"
#include <arrow/compute/api.h>
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
struct UnaryOpsTest : testing::Test {};
template <typename T>
struct RoundTest : testing::Test {};

TYPED_TEST_SUITE(UnaryOpsTest, NumericTypes);
TYPED_TEST_SUITE(RoundTest, FloatTypes);

namespace {
bool skip(std::string const& op, legate::Type const& type)
{
  std::set<std::string> float_ops = {"sin",
                                     "cos",
                                     "tan",
                                     "asin",
                                     "acos",
                                     "atan",
                                     "sinh",
                                     "cosh",
                                     "tanh",
                                     "asinh",
                                     "acosh",
                                     "atanh",
                                     "exp",
                                     "ln",
                                     "sqrt",
                                     "ceil",
                                     "floor",
                                     "round"};
  // Only invert op is supported for bools
  if (type.code() == legate::Type::Code::BOOL && op != "invert") { return true; }
  if (type.code() != legate::Type::Code::BOOL && op == "invert") { return true; }

  // negate op not supported for unsigned
  if (type.to_string().find("uint") != std::string::npos && op == "negate") { return true; }

  std::set<legate::Type::Code> float_types = {legate::Type::Code::FLOAT32,
                                              legate::Type::Code::FLOAT64,
                                              legate::Type::Code::FLOAT16,
                                              legate::Type::Code::COMPLEX64,
                                              legate::Type::Code::COMPLEX128};
  if (float_ops.count(op) && !float_types.count(type.code())) { return true; }

  // Bitwise op doesn't work for floats
  if (float_types.count(type.code()) && op == "bit_wise_not") { return true; }

  return false;
}
}  // namespace

void CompareArrow(const LogicalColumn& col, const std::vector<std::string>& unary_ops)
{
  for (const auto& op : unary_ops) {
    if (skip(op, col.type())) { continue; }

    auto expected = (*arrow::compute::CallFunction(op, {col.get_arrow()})).make_array();
    auto result   = unary_operation(col, op).get_arrow();

    // For integers check exact equality, for floats check approximate equality
    std::set<legate::Type::Code> float_types = {legate::Type::Code::FLOAT32,
                                                legate::Type::Code::FLOAT64,
                                                legate::Type::Code::FLOAT16,
                                                legate::Type::Code::COMPLEX64,
                                                legate::Type::Code::COMPLEX128};
    if (float_types.count(col.type().code())) {
      arrow::EqualOptions options;
      EXPECT_TRUE(expected->ApproxEquals(*result, options.nans_equal(true)))
        << "Failed for operation: " << op << " Input: " << col.repr()
        << " Expected: " << expected->ToString() << " Result: " << result->ToString();
    } else {
      EXPECT_TRUE(expected->Equals(*result))
        << "Failed for operation: " << op << " Input: " << col.repr()
        << " Expected: " << expected->ToString() << " Result: " << result->ToString();
    }
  }
}
std::vector<std::string> unary_ops = {"sin",   "cos",          "tan",    "asin",  "acos",  "atan",
                                      "sinh",  "cosh",         "tanh",   "asinh", "acosh", "atanh",
                                      "exp",   "ln",           "sqrt",   "ceil",  "floor", "abs",
                                      "round", "bit_wise_not", "invert", "negate"};

TYPED_TEST(UnaryOpsTest, UnaryOps)
{
  LogicalColumn column(narrow<TypeParam>({-1, -2, -3, -4}));
  CompareArrow(column, unary_ops);
}

TYPED_TEST(UnaryOpsTest, UnaryOpsWithNull)
{
  LogicalColumn column(narrow<TypeParam>({-1, -2, -3, -4}), {1, 0, 1, 0});
  CompareArrow(column, unary_ops);
}

TYPED_TEST(UnaryOpsTest, CastFromAnyToFloat32)
{
  LogicalColumn column(narrow<TypeParam>({-1, -2, -3, -4}));
  auto result = cast(column, cudf::data_type{cudf::type_id::FLOAT32});
  auto expected =
    ARROW_RESULT(arrow::compute::Cast(
                   column.get_arrow(), arrow::float32(), arrow::compute::CastOptions::Unsafe()))
      .make_array();
  EXPECT_TRUE(expected->Equals(*result.get_arrow()));
}

TYPED_TEST(UnaryOpsTest, CastFromInt16ToAny)
{
  LogicalColumn column(narrow<int16_t>({-1, -2, -3, -4}));
  auto to_dtype = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto expected = ARROW_RESULT(arrow::compute::Cast(column.get_arrow(),
                                                    to_arrow_type(to_dtype.id()),
                                                    arrow::compute::CastOptions::Unsafe()))
                    .make_array();
  LogicalColumn result = cast(column, to_dtype);
  EXPECT_TRUE(expected->Equals(*result.get_arrow()));
}

TYPED_TEST(RoundTest, RoundBasic)
{
  LogicalColumn column(
    narrow<double>({-1.5, -2.5, -3.5, -4.5, 1.5, 1.1234, 123455., -4321235., -1234.}));
  auto result = round(column, 0, "half_away_from_zero");
  auto expected =
    ARROW_RESULT(arrow::compute::Round(column.get_arrow(),
                                       arrow::compute::RoundOptions(
                                         0, arrow::compute::RoundMode::HALF_TOWARDS_INFINITY)))
      .make_array();
  EXPECT_TRUE(expected->Equals(*result.get_arrow()));

  result   = round(column, -1, "half_to_even");
  expected = ARROW_RESULT(arrow::compute::Round(column.get_arrow(),
                                                arrow::compute::RoundOptions(
                                                  -1, arrow::compute::RoundMode::HALF_TO_EVEN)))
               .make_array();
  EXPECT_TRUE(expected->Equals(*result.get_arrow()));
}
