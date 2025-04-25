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

#include <cudf/unary.hpp>

#include <legate_dataframe/core/column.hpp>

namespace legate::dataframe {

/**
 * @brief Cast column to a new data type.
 *
 * @param col Logical column as input
 * @param dtype The desired data type of the output column
 *
 * @returns Logical column of same size as `col` of the new data type
 */
LogicalColumn cast(const LogicalColumn& col, cudf::data_type dtype);

/**
 * @brief Performs unary operation on all values in column
 *
 * Note: For `decimal32` and `decimal64`, only `ABS`, `CEIL` and `FLOOR` are supported.
 *
 * @param col Logical column as input
 * @param op Operation to perform
 *
 * @returns Logical column of same size as `col` containing result of the operation
 */
LogicalColumn unary_operation(const LogicalColumn& col, cudf::unary_operator op);

}  // namespace legate::dataframe
