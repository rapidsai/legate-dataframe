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
const std::set<std::string> cudf_supported_binary_ops = {"add",
                                                         "divide",
                                                         "multiply",
                                                         "power",
                                                         "subtract",
                                                         "bit_wise_and",
                                                         "bit_wise_or",
                                                         "bit_wise_xor",
                                                         "shift_left",
                                                         "shift_right",
                                                         "logb",
                                                         "atan2",
                                                         "equal",
                                                         "greater",
                                                         "greater_equal",
                                                         "less",
                                                         "less_equal",
                                                         "not_equal",
                                                         // logical operators:
                                                         "and",
                                                         "or",
                                                         "and_kleene",
                                                         "or_kleene"};

class BinaryOpColColTask : public Task<BinaryOpColColTask, OpCode::BinaryOpColCol> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace task
/**
 * @brief Performs a binary operation between two columns.
 *
 * The output contains the result of `op(lhs[i], rhs[i])` for all `0 <= i < lhs.size()`.
 * Either or both columns may be "scalar" columns (e.g. created from a cudf scalar).
 * In that case, they will act as a scalar (identical to broadcasting them to all
 * entries of the second column).
 * If both are scalars, the result will also be marked as scalar.
 *
 * Regardless of the operator, the validity of the output value is the logical
 * AND of the validity of the two operands except NullMin and NullMax (logical OR).
 *
 * @param lhs         The left operand column
 * @param rhs         The right operand column
 * @param op          An arrow compute function - see arrow docs for supported operations e.g.
 * "add", "power"
 * @param output_type The desired data type of the output column
 * @return            Output column of `output_type` type containing the result of
 *                    the binary operation
 * @throw std::invalid_argument if operator not supported
 */
LogicalColumn binary_operation(const LogicalColumn& lhs,
                               const LogicalColumn& rhs,
                               std::string op,
                               std::shared_ptr<arrow::DataType> output_type);

}  // namespace legate::dataframe
