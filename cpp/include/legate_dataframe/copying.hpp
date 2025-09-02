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
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>

namespace legate::dataframe {

namespace task {

class CopyIfElseTask : public Task<CopyIfElseTask, OpCode::CopyIfElse> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace task
/**
 * @brief Performs a ternary assignment operation along the columns.
 *
 * The result will contain the value of `cond[i] ? lhs[i] : rhs[i]`.  Both `lhs` and `rhs`
 * may be scalar columns in which case they are broadcast against `cond`.
 * `lhs` and `rhs` must have the same type.
 *
 * @param cond The condition column to decide which value to copy
 * @param lhs  The left operand column
 * @param rhs  The right operand column
 * @return     The result of the ternary if_else operation.
 * @throw      std::invalid_argument if `lhs` and `rhs` have different types or `cond`
 *             is not boolean.
 */
LogicalColumn copy_if_else(const LogicalColumn& cond,
                           const LogicalColumn& lhs,
                           const LogicalColumn& rhs);

}  // namespace legate::dataframe
