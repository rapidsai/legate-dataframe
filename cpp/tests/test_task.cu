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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/filling.hpp>

using namespace legate::dataframe;

namespace {

static const char* library_name = "test.global_row_offset";

struct GlobalRowOffsetTask : public legate::LegateTask<GlobalRowOffsetTask> {
  static inline const auto TASK_CONFIG      = legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    auto tbl                               = argument::get_next_input<task::PhysicalTable>(ctx);
    auto output                            = argument::get_next_output<task::PhysicalColumn>(ctx);
    std::vector<task::PhysicalColumn> cols = tbl.release();
    int64_t offset                         = cols.at(0).global_row_offset();
    int64_t nrows                          = cols.at(0).num_rows();

    // We expect the columns of a table are aligned.
    EXPECT_EQ(offset, cols.at(1).global_row_offset());

    // Write our row offset and size
    cudf::test::fixed_width_column_wrapper<int64_t> out({offset, nrows});
    output.move_into(out.release());
  }
};

struct TaskArgumentMix : public legate::LegateTask<TaskArgumentMix> {
  static inline const auto TASK_CONFIG      = legate::TaskConfig{legate::LocalTaskID{1}};
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    // When mixing legate-dataframe and non-legate-dataframe arguments,
    // we use `get_task_argument_indices()` to get the start indices of the
    // non-legate-dataframe arguments.
    // NB: `get_task_argument_indices()` must be called *after* all the
    //     legate-dataframe arguments has been retrieved.

    TaskContext ctx{context};

    {
      auto [scalar_idx, input_idx, output_idx] = ctx.get_task_argument_indices();
      EXPECT_EQ(scalar_idx, 0);
      EXPECT_EQ(input_idx, 0);
      EXPECT_EQ(output_idx, 0);
    }
    const auto input = argument::get_next_input<task::PhysicalColumn>(ctx);
    {
      auto [scalar_idx, input_idx, output_idx] = ctx.get_task_argument_indices();
      EXPECT_EQ(scalar_idx, 1);
      EXPECT_EQ(input_idx, 1);
      EXPECT_EQ(output_idx, 0);
    }
    auto output = argument::get_next_output<task::PhysicalColumn>(ctx);
    {
      auto [scalar_idx, input_idx, output_idx] = ctx.get_task_argument_indices();
      EXPECT_EQ(scalar_idx, 3);
      EXPECT_EQ(input_idx, 1);
      EXPECT_EQ(output_idx, 1);
    }
    auto scalar = argument::get_next_scalar<int32_t>(ctx);
    {
      auto [scalar_idx, input_idx, output_idx] = ctx.get_task_argument_indices();
      EXPECT_EQ(scalar_idx, 4);
      EXPECT_EQ(input_idx, 1);
      EXPECT_EQ(output_idx, 1);
    }
    output.move_into(std::make_unique<cudf::column>(input.column_view()));
  }
};

legate::Library get_test_library()
{
  static bool prepared = false;
  auto runtime         = legate::Runtime::get_runtime();
  if (prepared) { return runtime->find_library(library_name); }
  prepared = true;

  auto context = runtime->create_library(library_name);
  GlobalRowOffsetTask::register_variants(context);
  TaskArgumentMix::register_variants(context);
  return context;
}

void check_global_row_offset(LogicalTable& input)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task = runtime->create_task(get_test_library(), GlobalRowOffsetTask::TASK_CONFIG.task_id());

  // Launch task
  LogicalColumn res = LogicalColumn::empty_like(arrow::int64(), /* nullable = */ false);
  argument::add_next_input(task, input);
  argument::add_next_output(task, res);
  argument::add_next_scalar(task, input.num_rows());
  runtime->submit(std::move(task));

  // Check result
  legate::PhysicalArray ary = res.get_physical_array();
  auto acc                  = ary.data().read_accessor<int64_t, 1>();
  int64_t expected_offset   = acc[0] + acc[1];
  EXPECT_EQ(acc[0], 0);
  for (long unsigned int i = 2; i < res.num_rows() - 1; i += 2) {
    EXPECT_EQ(acc[i], expected_offset);
    expected_offset = acc[i] + acc[i + 1];
  }
}

}  // namespace

TEST(TaskTest, GlobalRowOffset)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<int64_t> a({0, 1, 2, 3, 4, 5});
  cudf::test::strings_column_wrapper b({"0", "1", "2", "3", "4", "5"});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table _tbl(std::move(cols));
  LogicalTable input(_tbl, {"a", "b"});
  check_global_row_offset(input);
}

TEST(TaskTest, GlobalRowOffsetSingleRow)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<int64_t> a({5});
  cudf::test::strings_column_wrapper b({"5"});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table _tbl(std::move(cols));
  LogicalTable input(_tbl, {"a", "b"});
  check_global_row_offset(input);
}

TEST(TaskTest, GlobalRowOffsetEmpty)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<int64_t> a({});
  cudf::test::fixed_width_column_wrapper<int64_t> b({});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table _tbl(std::move(cols));
  LogicalTable input(_tbl, {"a", "b"});
  check_global_row_offset(input);
}

TEST(TaskTest, TaskArgumentMix)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(get_test_library(), TaskArgumentMix::TASK_CONFIG.task_id());

  auto input     = sequence(100, 0);
  auto output    = LogicalColumn::empty_like(input);
  int32_t scalar = 42;

  argument::add_next_input(task, input);
  argument::add_next_output(task, output);
  argument::add_next_scalar(task, scalar);
  runtime->submit(std::move(task));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output.get_cudf()->view(), input.get_cudf()->view());
}
