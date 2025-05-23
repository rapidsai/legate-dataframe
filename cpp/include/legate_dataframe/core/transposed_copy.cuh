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

#include <cudf/table/table_view.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe {

void copy_into_tranposed(GPUTaskContext& ctx,
                         legate::PhysicalArray& array,
                         cudf::table_view tbl,
                         size_t offset,
                         legate::Scalar& null_value);

}  // namespace legate::dataframe
