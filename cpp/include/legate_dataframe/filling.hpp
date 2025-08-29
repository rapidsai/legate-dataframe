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

#pragma once

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>

namespace legate::dataframe {

namespace task {

class SequenceTask : public Task<SequenceTask, OpCode::Sequence> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace task

/**
 * @brief Fills a column with a sequence of int64 values
 *
 * Creates a new column and fills with @p size values starting at @p init and
 * incrementing by 1, generating the sequence
 * [ init, init+1, init+2, ... init + (size - 1)]
 *
 * ```
 * size = 3
 * init = 0
 * return = [0, 1, 2]
 * ```
 *
 * Notice, this is primarily for C++ testing and examples for now. TODO: implement
 * all of the cudf features <https://github.com/rapidsai/legate-dataframe/issues/74>
 *
 * @param size Size of the output column
 * @param init First value in the sequence
 * @return The result column (int64) containing the generated sequence
 */
LogicalColumn sequence(size_t size, int64_t init);

}  // namespace legate::dataframe
