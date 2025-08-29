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

#pragma once

#include <legate.h>
#include <legate/mapping/mapping.h>

namespace legate::dataframe {

auto GetLogger() -> Legion::Logger&;

namespace task {

namespace OpCode {
enum : int {
  ApplyBooleanMask,
  CSVWrite,
  CSVRead,
  Cast,
  CopyIfElse,
  ParquetWrite,
  ParquetRead,
  ParquetReadArray,
  ReplaceNullsWithScalar,
  Round,
  UnaryOp,
  BinaryOpColCol,
  BinaryOpColScalar,
  BinaryOpScalarCol,
  Join,
  JoinConcurrent,
  ToTimestamps,
  ExtractTimestampComponent,
  ReduceLocal,
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
  using Registrar                      = Registry;
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{ID}};
};

}  // namespace task

legate::Library& get_library();

bool get_prefer_eager_allocations();

}  // namespace legate::dataframe
