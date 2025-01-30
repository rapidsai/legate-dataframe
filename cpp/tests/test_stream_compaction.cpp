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

#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/stream_compaction.hpp>

using namespace legate::dataframe;
template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;
using CVector        = std::vector<std::unique_ptr<cudf::column>>;

TEST(StreamCompactionTest, ApplyBooleanMask)
{
  column_wrapper<int32_t> col_0{{5, 4, 3, 1, 2, 0}};
  strcol_wrapper col_1({"this", "is", "a", "string", "column", "!"});
  column_wrapper<double> col_2{{0, 1, 2, 3, 4, 5}};

  column_wrapper<bool> boolean_mask{{true, false, true, true, false, false}};

  CVector cols;
  cols.push_back(col_0.release());
  cols.push_back(col_1.release());
  cols.push_back(col_2.release());

  cudf::table tbl(std::move(cols));

  LogicalTable lg_tbl{tbl.view(), {"a", "b", "c"}};

  auto expect = cudf::apply_boolean_mask(tbl, boolean_mask);
  auto result = apply_boolean_mask(lg_tbl, LogicalColumn{boolean_mask});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, result.get_cudf()->view());

  // Additionally, check that a null mask is honored by legate-dataframe
  column_wrapper<bool> boolean_mask_nulls{{true, false, true, true, false, false},
                                          {true, true, true, false, false, true}};

  expect = cudf::apply_boolean_mask(tbl, boolean_mask_nulls);
  result = apply_boolean_mask(lg_tbl, LogicalColumn{boolean_mask_nulls});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expect, result.get_cudf()->view());
}
