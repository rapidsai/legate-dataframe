/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation.hpp>

#include <legate_dataframe/core/column.hpp>

namespace legate::dataframe {

/**
 * @brief Reduce a column given a libcudf reduction.
 *
 * @param col Logical column to reduce
 * @param agg The reduction to perform
 * @param output_dtype The output cudf dtype.
 * @param initial Optional initial scalar (column) only supported for numeric types.
 * @returns A LogicalColumn marked as scalar.
 */
LogicalColumn reduce(
  const LogicalColumn& col,
  const cudf::reduce_aggregation& agg,
  cudf::data_type output_dtype,
  std::optional<std::reference_wrapper<const LogicalColumn>> initial = std::nullopt);

}  // namespace legate::dataframe
