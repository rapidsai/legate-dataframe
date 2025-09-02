/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <legate.h>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>

namespace legate::dataframe {

namespace task {

class ContainsTask : public Task<ContainsTask, OpCode::Contains> {
 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace task

/**
 * @brief Check if haystack contains the values in needles.
 *
 * The result will contain boolean values indicating whether each element in the input
 * column exists in the set of values.
 *
 * @param haystack Column of values to search against. This column is currently broadcast to all
 * workers and assumed to be small.
 * @param needles Column of values to check if they exist in the haystack.
 * @return Boolean column indicating which values exist in the set, has the same
 * size and nullability as haystack.
 * @throw std::invalid_argument if the types are incompatible
 */
LogicalColumn contains(const LogicalColumn& values, const LogicalColumn& set_values);

}  // namespace legate::dataframe
