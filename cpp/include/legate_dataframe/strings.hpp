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
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>

namespace legate::dataframe {

namespace task {

class StringMatchesTask : public Task<StringMatchesTask, OpCode::StringMatches> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_DATAFRAME_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace task

namespace strings {

/**
 * @brief Checks if strings match a given pattern.
 *
 * @param match_func The function to use "starts_with", "ends_with", "match_substring",
 * or "match_substring_regex".
 * (Note that the "match_substring*" check for containment not full matches.)
 * @param column The column of string values to check
 * @param pattern The pattern string to check for
 * @return A boolean column indicating which values match the pattern
 */
LogicalColumn match(const std::string& match_func,
                    const LogicalColumn& column,
                    const std::string& pattern);

}  // namespace strings

}  // namespace legate::dataframe
