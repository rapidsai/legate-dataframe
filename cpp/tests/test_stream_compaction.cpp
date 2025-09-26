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

#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <legate_dataframe/stream_compaction.hpp>

using namespace legate::dataframe;

TEST(StreamCompactionTest, ApplyBooleanMask)
{
  LogicalColumn col_0(std::vector<int32_t>{5, 4, 3, 1, 2, 0});
  LogicalColumn col_1(std::vector<std::string>{"this", "is", "a", "string", "column", "!"});
  LogicalColumn col_2(std::vector<double>{0, 1, 2, 3, 4, 5});

  LogicalColumn boolean_mask(std::vector<bool>{true, false, true, true, false, false});

  LogicalTable tbl{{col_0, col_1, col_2}, {"a", "b", "c"}};

  auto result   = apply_boolean_mask(tbl, boolean_mask);
  auto expected = ARROW_RESULT(arrow::compute::CallFunction(
                                 "filter", {tbl.get_arrow(), boolean_mask.get_arrow()}))
                    .table();
  EXPECT_TRUE(result.get_arrow()->Equals(*expected));

  // Additionally, check that a null mask is honored by legate-dataframe
  LogicalColumn boolean_mask_nulls(std::vector<bool>{true, false, true, true, false, false},
                                   {true, true, true, false, false, true});

  result   = apply_boolean_mask(tbl, boolean_mask_nulls);
  expected = ARROW_RESULT(arrow::compute::CallFunction(
                            "filter", {tbl.get_arrow(), boolean_mask_nulls.get_arrow()}))
               .table();
  EXPECT_TRUE(result.get_arrow()->Equals(*expected));
}

TEST(StreamCompactionTest, Distinct)
{
  LogicalColumn col_0(std::vector<int32_t>{0, 0, 0, 1, 1, 1},
                      {false, false, false, true, true, true});
  LogicalColumn col_1(std::vector<std::string>{"this", "this", "string", "string", "col", "col"});
  LogicalColumn col_2(std::vector<double>{0, 0, 2, 3, 4, 4});

  LogicalTable tbl{{col_0, col_1, col_2}, {"a", "b", "c"}};

  auto result = distinct(tbl, {"a", "b"});

  // Hardcoded expected result (we have no simple arrow unique call).
  LogicalColumn col_0_exp(std::vector<int32_t>{0, 0, 1, 1}, {false, false, true, true});
  LogicalColumn col_1_exp(std::vector<std::string>{"this", "string", "string", "col"});
  LogicalColumn col_2_exp(std::vector<double>{0, 2, 3, 4});
  LogicalTable expected{{col_0_exp, col_1_exp, col_2_exp}, {"a", "b", "c"}};
  auto expected_arrow = expected.get_arrow();

  EXPECT_TRUE(result.get_arrow()->Equals(*expected_arrow));
}
