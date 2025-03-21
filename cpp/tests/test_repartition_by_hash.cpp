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

#include <legate_dataframe/parquet.hpp>
#include <legate_dataframe/utils.hpp>

#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>

#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

using namespace legate::dataframe;

namespace {

static const char* library_name = "test.repartition_by_hash";

struct CheckHash : public legate::LegateTask<CheckHash> {
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);

  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    const auto table      = argument::get_next_input<task::PhysicalTable>(ctx);
    auto result           = argument::get_next_output<task::PhysicalTable>(ctx);
    const auto table_keys = argument::get_next_scalar_vector<int32_t>(ctx);
    std::unique_ptr<cudf::table> cudf_result =
      task::repartition_by_hash(ctx, table.table_view(), table_keys);

    auto [partition_table, partition_offsets] = cudf::hash_partition(cudf_result->view(),
                                                                     table_keys,
                                                                     ctx.nranks,
                                                                     cudf::hash_id::HASH_MURMUR3,
                                                                     cudf::DEFAULT_HASH_SEED,
                                                                     ctx.stream(),
                                                                     ctx.mr());
    result.move_into(std::move(cudf_result));

    // Check that the partition assigned to our rank gets all the rows.
    partition_offsets.push_back(partition_table->num_rows());
    EXPECT_EQ(partition_offsets[ctx.rank + 1] - partition_offsets[ctx.rank],
              partition_table->num_rows());
    // Check that the other partitions are empty.
    for (size_t i = 0; i < partition_offsets.size() - 1; ++i) {
      if (i != static_cast<size_t>(ctx.rank)) {
        EXPECT_EQ(partition_offsets[i] == partition_offsets[i + 1], true);
      }
    }
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  CheckHash::register_variants(context);
}
}  // namespace
TEST(RepartitionByHash, NoNull)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, CheckHash::TASK_CONFIG.task_id());

  // Create a three column cudf and legate table
  cudf::test::fixed_width_column_wrapper<int32_t> col0{{3, 1, 2, 0, 2}};
  cudf::test::strings_column_wrapper col1({"s1", "s1", "s0", "s4", "s0"});
  cudf::test::fixed_width_column_wrapper<float> col2{{0, 1, 2, 3, 4}};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cudf::table cudf_tbl(std::move(cols));
  LogicalTable lg_in(cudf_tbl, std::vector<std::string>({"a", "b", "c"}));
  LogicalTable lg_out = LogicalTable::empty_like(lg_in);

  // Launch task
  argument::add_next_input(task, lg_in);
  argument::add_next_output(task, lg_out);
  argument::add_next_scalar_vector(task, std::vector<int32_t>{0, 1});
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  // Compare the input column with the result. We ignore the row order by sorting them first.
  auto expect      = cudf::gather(cudf_tbl.view(), cudf::sorted_order(cudf_tbl.view())->view());
  auto cudf_lg_out = lg_out.get_cudf();
  auto lg_out_sort_order = cudf::sorted_order(cudf_lg_out->view());
  auto result = cudf::gather(cudf_lg_out->view(), cudf::sorted_order(cudf_lg_out->view())->view());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expect->view());
}

TEST(RepartitionByHash, WithNullInNumericColumn)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, CheckHash::TASK_CONFIG.task_id());

  // Create a three column cudf and legate table
  cudf::test::fixed_width_column_wrapper<int32_t> col0{{3, 1, 2, 0, 2}, {1, 0, 1, 1, 1}};
  cudf::test::strings_column_wrapper col1({"s1", "s1", "s0", "s4", "s0"});
  cudf::test::fixed_width_column_wrapper<float> col2{{0, 1, 2, 3, 4}};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cudf::table cudf_tbl(std::move(cols));
  LogicalTable lg_in(cudf_tbl, std::vector<std::string>({"a", "b", "c"}));
  LogicalTable lg_out = LogicalTable::empty_like(lg_in);

  // Launch task
  argument::add_next_input(task, lg_in);
  argument::add_next_output(task, lg_out);
  argument::add_next_scalar_vector(task, std::vector<int32_t>{0, 1});
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  // Compare the input column with the result. We ignore the row order by sorting them first.
  auto expect      = cudf::gather(cudf_tbl.view(), cudf::sorted_order(cudf_tbl.view())->view());
  auto cudf_lg_out = lg_out.get_cudf();
  auto lg_out_sort_order = cudf::sorted_order(cudf_lg_out->view());
  auto result = cudf::gather(cudf_lg_out->view(), cudf::sorted_order(cudf_lg_out->view())->view());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expect->view());
}

TEST(RepartitionByHash, WithNullInStringsColumn)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, CheckHash::TASK_CONFIG.task_id());

  // Create a three column cudf and legate table
  cudf::test::fixed_width_column_wrapper<int32_t> col0{{3, 1, 2, 0, 2}};
  cudf::test::strings_column_wrapper col1({"s1", "s1", "s0", "s4", "s0"}, {1, 0, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<float> col2{{0, 1, 2, 3, 4}};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(col0.release());
  cols.push_back(col1.release());
  cols.push_back(col2.release());
  cudf::table cudf_tbl(std::move(cols));
  LogicalTable lg_in(cudf_tbl, std::vector<std::string>({"a", "b", "c"}));
  LogicalTable lg_out = LogicalTable::empty_like(lg_in);

  // Launch task
  argument::add_next_input(task, lg_in);
  argument::add_next_output(task, lg_out);
  argument::add_next_scalar_vector(task, std::vector<int32_t>{0, 1});
  task.add_communicator("nccl");
  runtime->submit(std::move(task));

  // Compare the input column with the result. We ignore the row order by sorting them first.
  auto expect      = cudf::gather(cudf_tbl.view(), cudf::sorted_order(cudf_tbl.view())->view());
  auto cudf_lg_out = lg_out.get_cudf();
  auto lg_out_sort_order = cudf::sorted_order(cudf_lg_out->view());
  auto result = cudf::gather(cudf_lg_out->view(), cudf::sorted_order(cudf_lg_out->view())->view());

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result->view(), expect->view());
}
