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

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <legate.h>

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/groupby_aggregation.hpp>

#include <iostream>

namespace legate::dataframe {
namespace task {

class HashPartitionTask : public Task<HashPartitionTask, OpCode::HashPartition> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};
    auto table      = argument::get_next_input<PhysicalTable>(ctx);
    auto num_parts  = argument::get_next_scalar<int>(ctx);
    auto keys_idx   = argument::get_next_scalar_vector<cudf::size_type>(ctx);

    auto partitions = ctx.get_next_output_arg();
    auto output     = argument::get_next_output<PhysicalTable>(ctx);

    auto table_view = table.table_view();
    auto row_offset = table.global_row_offset();

    if (num_parts < 0) {
      num_parts = ctx.nranks;  // default to same number as initially.
    }

    std::unique_ptr<cudf::table> partition_table;
    std::vector<cudf::size_type> partition_starts(num_parts);
    if (table_view.num_rows() == 0) {
      partition_table = cudf::empty_like(table_view);
      std::iota(partition_starts.begin(), partition_starts.end(), 0);
    } else {
      auto res = cudf::hash_partition(table_view,
                                      keys_idx,
                                      num_parts,
                                      cudf::hash_id::HASH_MURMUR3,
                                      cudf::DEFAULT_HASH_SEED,
                                      ctx.stream(),
                                      ctx.mr());
      partition_table.swap(res.first);
      partition_starts.swap(res.second);
      partition_starts.push_back(table_view.num_rows());
    }

    auto partitions_buf = partitions.data().create_output_buffer<legate::Rect<1>, 2>(
      {1, num_parts}, true);

    std::cout << "Target?: " << partitions.data().target() << std::endl;

    for (int i = 0; i < num_parts; i++) {
      // legate ranges are inclusive:
      partitions_buf[{0, i}].lo = row_offset + partition_starts.at(i);
      partitions_buf[{0, i}].hi = row_offset + partition_starts.at(i+1) - 1;
    }

    output.move_into(std::move(partition_table));
  }
};

}  // namespace task


std::pair<LogicalTable, legate::LogicalArray>
hashpartition(const LogicalTable& table, const std::set<size_t>& keys, int num_parts)
{
  if (keys.size() == 0) {
    throw std::invalid_argument("keys must have at least one entry.");
  }
  if (num_parts < 1 && num_parts != -1) {
    throw std::invalid_argument("num_parts must be >=1 or -1 to indicate same as input.");
  }

  LogicalTable output = LogicalTable::empty_like(table);

  // The result of this task are "partitions" described in a nranks x num_parts
  // array of partition ranges.  Use an unbound store so we can avoid defining
  // the number of parts here.
  // TODO: an unbound store should be unnecessary here!
  legate::LogicalArray partitions(
    legate::Runtime::get_runtime()->create_array(legate::rect_type(1), 2)
  );

  auto runtime = legate::Runtime::get_runtime();
  legate::AutoTask task =
    runtime->create_task(get_library(), task::HashPartitionTask::TASK_ID);
  argument::add_next_input(task, table);
  argument::add_next_scalar(task, num_parts);
  argument::add_next_scalar_vector(task, std::vector<cudf::size_type>(keys.begin(), keys.end()));

  task.add_output(partitions);
  argument::add_next_output(task, output);

  runtime->submit(std::move(task));
  return std::make_pair(output, partitions);
}


std::pair<LogicalTable, legate::LogicalArray>
hashpartition(const LogicalTable& table, const std::set<std::string>& keys, int num_parts)
{
  std::set<size_t> key_idx;
  auto colname_map = table.get_column_names();
  for (auto key : keys) {
    key_idx.insert(colname_map.at(key));
  }

  return hashpartition(table, key_idx, num_parts);
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::HashPartitionTask::register_variants();
}

}  // namespace
