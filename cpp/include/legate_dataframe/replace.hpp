/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <legate_dataframe/core/column.hpp>

namespace legate::dataframe {

/**
 * @brief Replace nulls in a column with a scalar
 *
 *
 * @param col Logical column as input.
 * @param scalar value to fill with. Must be a LogicalColumn marked as scalar.
 *
 * @returns Logical column where nulls have been replaced with `value`
 */
LogicalColumn replace_nulls(const LogicalColumn& col, const LogicalColumn& value);

}  // namespace legate::dataframe
