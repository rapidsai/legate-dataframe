/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#pragma once

#include <legate.h>
#include <legate/mapping/mapping.h>

namespace legate::dataframe {
namespace task {

namespace OpCode {
enum : int {
  CSVWrite,
  CSVRead,
  ParquetWrite,
  ParquetRead,
  ReplaceNullsWithScalar,
  UnaryOp,
  BinaryOpColCol,
  BinaryOpColScalar,
  BinaryOpScalarCol,
  Join,
  ToTimestamps,
  ExtractTimestampComponent,
  Sequence,
  Sort,
  GroupByAggregation
};
}

struct Registry {
  static legate::TaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar               = Registry;
  static constexpr auto TASK_ID = legate::LocalTaskID{ID};
};

}  // namespace task

legate::Library& get_library();

}  // namespace legate::dataframe
