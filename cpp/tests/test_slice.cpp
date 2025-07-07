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
#include <arrow/api.h>

#include <cudf/column/column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

using namespace legate::dataframe;

TEST(SliceTest, SliceColumns)
{
  LogicalColumn col(narrow<int32_t>({1, 2, 3, 4, 5}), {1, 0, 1, 0, 1});

  auto result = col.slice(legate::Slice(0, 5));
  auto expect = col.get_arrow()->Slice(0, 5);  // uses start, length
  EXPECT_TRUE(expect->Equals(*result.get_arrow()));

  result = col.slice(legate::Slice(1, 3));
  expect = col.get_arrow()->Slice(1, 2);
  EXPECT_TRUE(expect->Equals(*result.get_arrow()));

  result = col.slice(legate::Slice(0, -2));
  expect = col.get_arrow()->Slice(0, 3);  // uses start, length
  EXPECT_TRUE(expect->Equals(*result.get_arrow()));
}
