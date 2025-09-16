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
#include <legate_dataframe/strings.hpp>

using namespace legate::dataframe;

void CompareArrow(const std::string& match_func,
                  const LogicalColumn& column,
                  const std::string& pattern)
{
  std::vector<arrow::Datum> args = {column.get_arrow()};
  auto options                   = arrow::compute::MatchSubstringOptions(pattern);
  auto expected =
    (ARROW_RESULT(arrow::compute::CallFunction(match_func, args, &options))).make_array();
  auto result = strings::match(match_func, column, pattern).get_arrow();

  EXPECT_TRUE(expected->Equals(*result))
    << "Failed " << match_func << ": " << column.get_arrow()->ToString() << " pattern: " << pattern
    << " Expected: " << expected->ToString() << " Result: " << result->ToString();
}

TEST(StringsTest, StartsWithBasic)
{
  LogicalColumn values(std::vector<std::string>{"hello", "world", "help", "spam"});
  CompareArrow("starts_with", values, "hel");  // codespell:ignore
}

TEST(StringsTest, EndsWithBasic)
{
  LogicalColumn values(std::vector<std::string>{"hello", "", "help", "fello"});
  CompareArrow("ends_with", values, "llo");  // codespell:ignore
}

TEST(StringsTest, Contains)
{
  LogicalColumn values(std::vector<std::string>{"hord", "world", "help"});
  CompareArrow("match_substring", values, "or");
}

TEST(StringsTest, ContainsRegex)
{
  LogicalColumn values(std::vector<std::string>{"hord", "world", "help"});
  CompareArrow("match_substring_regex", values, "o.+d");
}

TEST(StringsTest, MatchesWithNulls)
{
  LogicalColumn values(std::vector<std::string>{"lolo", "lolo", "spam", "spam"},
                       std::vector<bool>{true, false, true, false});
  CompareArrow("starts_with", values, "lo");
  CompareArrow("ends_with", values, "lo");
  CompareArrow("match_substring", values, "lo");
  CompareArrow("match_substring_regex", values, "lo");
}

TEST(StringsTest, ErrorNonStringColumn)
{
  LogicalColumn values(std::vector<int>{1, 2, 3});

  EXPECT_THROW(strings::match("starts_with", values, "he"), std::invalid_argument);
}
