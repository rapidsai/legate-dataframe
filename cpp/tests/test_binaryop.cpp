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
#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

using namespace legate::dataframe;

template <typename T>
struct BinaryOpsTest : public testing::Test {};

// Edge cases for unsupported operations or inconsistent implementations between cudf/arrow
bool skip(std::string const& op, legate::Type const& type)
{
  std::set<std::string> arithmetic_ops = {"add", "subtract", "multiply", "divide", "power"};
  std::set<std::string> float_ops      = {"atan2", "logb"};
  std::set<std::string> bitwise_ops    = {
    "bit_wise_and", "bit_wise_or", "bit_wise_xor", "shift_left", "shift_right"};

  bool use_cudf =
    legate::Runtime::get_runtime()->get_machine().count(legate::mapping::TaskTarget::GPU) > 0;
  if (use_cudf) {
    // cudf does something different overflow
    // Avoid these operations on small types
    std::set<std::string> overflow_ops          = {"shift_left", "shift_right", "power"};
    std::set<legate::Type::Code> small_integers = {legate::Type::Code::INT8,
                                                   legate::Type::Code::INT16,
                                                   legate::Type::Code::UINT8,
                                                   legate::Type::Code::UINT16};
    if (overflow_ops.count(op) && small_integers.count(type.code())) { return true; }
  }

  // Arithmetic or bitwise operations on bools are not supported
  if (type.code() == legate::Type::Code::BOOL) {
    if (arithmetic_ops.count(op)) { return true; }
    if (bitwise_ops.count(op)) { return true; }
  }

  // Atan2 and logb dont make much sense on integers
  std::set<legate::Type::Code> float_types = {legate::Type::Code::FLOAT32,
                                              legate::Type::Code::FLOAT64,
                                              legate::Type::Code::FLOAT16,
                                              legate::Type::Code::COMPLEX64,
                                              legate::Type::Code::COMPLEX128};
  if (float_ops.count(op) && !float_types.count(type.code())) { return true; }

  // We don't support bitwise operations on floating point types
  if (bitwise_ops.count(op) && float_types.count(type.code())) { return true; }

  return false;
}

void CompareArrow(const LogicalColumn& lhs,
                  const LogicalColumn& rhs,
                  const std::vector<std::string>& binary_ops)
{
  for (const auto& op : binary_ops) {
    if (skip(op, lhs.type()) || skip(op, rhs.type())) { continue; }
    std::vector<arrow::Datum> args(2);
    if (lhs.is_scalar()) {
      auto scalar = ARROW_RESULT(lhs.get_arrow()->GetScalar(0));
      args[0]     = scalar;
    } else {
      args[0] = lhs.get_arrow();
    }
    if (rhs.is_scalar()) {
      auto scalar = ARROW_RESULT(rhs.get_arrow()->GetScalar(0));
      args[1]     = scalar;
    } else {
      args[1] = rhs.get_arrow();
    }

    // Specify bool output for equality comparisons
    std::set<std::string> equality_ops = {
      "equal", "greater", "greater_equal", "less", "less_equal", "not_equal"};
    auto output_type = lhs.arrow_type();
    if (equality_ops.count(op)) { output_type = arrow::boolean(); }

    auto expected = (*arrow::compute::CallFunction(op, args)).make_array();
    expected      = ARROW_RESULT(
                 arrow::compute::Cast(expected, output_type, arrow::compute::CastOptions::Unsafe()))
                 .make_array();
    /*
std::cout << op << std::endl;
std::cout << "LHS type: " << lhs.arrow_type()->ToString() << std::endl;
std::cout << "LHS type after arrow conversion " << lhs.get_arrow()->type()->ToString() << std::endl;
std::cout << "RHS type: " << rhs.arrow_type()->ToString() << std::endl;
*/

    auto result = binary_operation(lhs, rhs, op, output_type).get_arrow();

    // For integers check exact equality, for floats check approximate equality
    std::set<legate::Type::Code> float_types = {legate::Type::Code::FLOAT32,
                                                legate::Type::Code::FLOAT64,
                                                legate::Type::Code::FLOAT16,
                                                legate::Type::Code::COMPLEX64,
                                                legate::Type::Code::COMPLEX128};
    if (float_types.count(lhs.type().code())) {
      arrow::EqualOptions options;
      EXPECT_TRUE(expected->ApproxEquals(*result, options.nans_equal(true)))
        << "Failed for operation: " << op << " LHS: " << lhs.repr() << " RHS: " << rhs.repr()
        << " Expected: " << expected->ToString() << " Result: " << result->ToString();
    } else {
      EXPECT_TRUE(expected->Equals(*result))
        << "Failed for operation: " << op << " LHS: " << lhs.repr() << " RHS: " << rhs.repr()
        << " Expected: " << expected->ToString() << " Result: " << result->ToString();
    }
  }
}

using NumericTypesWithoutBool = ::testing::Types<int8_t,
                                                 int16_t,
                                                 int32_t,
                                                 int64_t,
                                                 uint8_t,
                                                 uint16_t,
                                                 uint32_t,
                                                 uint64_t,
                                                 float,
                                                 double,
                                                 bool>;

TYPED_TEST_SUITE(BinaryOpsTest, NumericTypesWithoutBool);

std::vector<std::string> ops = {"add",
                                "subtract",
                                "multiply",
                                "divide",
                                "power",
                                "logb",
                                "atan2",
                                "equal",
                                "greater",
                                "greater_equal",
                                "less",
                                "less_equal",
                                "not_equal",
                                "bit_wise_and",
                                "bit_wise_or",
                                "bit_wise_xor",
                                "shift_left",
                                "shift_right"};

TYPED_TEST(BinaryOpsTest, ColCol)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}));
  LogicalColumn rhs(narrow<TypeParam>({5, 6, 7, 8}));
  CompareArrow(lhs, rhs, ops);
}

TYPED_TEST(BinaryOpsTest, ColColWithNull)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}), {1, 0, 1, 0});
  LogicalColumn rhs(narrow<TypeParam>({5, 6, 7, 8}), {1, 0, 1, 0});
  CompareArrow(lhs, rhs, ops);
}

TYPED_TEST(BinaryOpsTest, ColScalar)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}));
  LogicalColumn rhs(narrow<TypeParam>({1}), {}, true);
  CompareArrow(lhs, rhs, ops);
}

TYPED_TEST(BinaryOpsTest, ColScalarWithNull)
{
  LogicalColumn lhs(narrow<TypeParam>({1, 2, 3, 4}), {1, 0, 1, 0});
  LogicalColumn rhs(narrow<TypeParam>({1}), {1}, true);
  CompareArrow(lhs, rhs, ops);
}

TYPED_TEST(BinaryOpsTest, ScalarCol)
{
  LogicalColumn lhs(narrow<TypeParam>({1}), {}, true);
  LogicalColumn rhs(narrow<TypeParam>({1, 2, 3, 4}));
  CompareArrow(lhs, rhs, ops);
}

TEST(BinaryOpsTest, BadOp)
{
  LogicalColumn lhs(narrow<int32_t>({1, 2, 3, 4}));
  LogicalColumn rhs(narrow<int32_t>({5, 6, 7, 8}));
  EXPECT_THAT([=]() { binary_operation(lhs, rhs, "bad_op", lhs.arrow_type()); },
              testing::ThrowsMessage<std::invalid_argument>(testing::HasSubstr("operator")));
}
