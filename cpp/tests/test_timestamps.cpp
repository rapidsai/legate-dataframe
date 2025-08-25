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

#include <legate.h>

#include "test_utils.hpp"
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/timestamps.hpp>

using namespace legate::dataframe;

TEST(TimestampsTest, ToTimestamps)
{
  std::vector<arrow::TimeUnit::type> time_units = {
    arrow::TimeUnit::SECOND, arrow::TimeUnit::MILLI, arrow::TimeUnit::MICRO, arrow::TimeUnit::NANO};

  LogicalColumn input({"2010-06-19T13:55", "2011-06-19T13:55", "", "2010-07-19T13:55"},
                      {1, 1, 0, 0});

  std::string format{"%Y-%m-%dT%H:%M"};

  for (auto& unit : time_units) {
    auto timestamp_type = arrow::timestamp(unit);
    auto result         = to_timestamps(input, timestamp_type, format);
    arrow::compute::StrptimeOptions options(format, unit);
    auto expected =
      ARROW_RESULT(arrow::compute::CallFunction("strptime", {input.get_arrow()}, &options))
        .make_array();
    EXPECT_TRUE(expected->Equals(result.get_arrow()));
  }
}

TEST(TimestampsTest, ExtractTimestampComponent)
{
  LogicalColumn input({"2010-06-19T13:55", "2011-06-19T13:55", "", "2010-07-19T13:55"},
                      {1, 1, 0, 0});
  std::string format{"%Y-%m-%dT%H:%M"};

  auto timestamps = to_timestamps(input, arrow::timestamp(arrow::TimeUnit::SECOND), format);
  std::vector<std::string> components = {
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
  };
  for (auto component : components) {
    auto result = extract_timestamp_component(timestamps, component);
    auto expected =
      ARROW_RESULT(arrow::compute::CallFunction(component, {timestamps.get_arrow()})).make_array();
    EXPECT_TRUE(expected->Equals(result.get_arrow()));
  }
}

TEST(TimestampsTest, ExtractBadTimestamp)
{
  LogicalColumn input(narrow<int>({1, 2, 3, 4}), {1, 1, 0, 0});

  EXPECT_THROW(extract_timestamp_component(input, "year"), std::invalid_argument);
}
