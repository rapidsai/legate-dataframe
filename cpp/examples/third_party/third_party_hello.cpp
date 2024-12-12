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

#include <legate.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/filling.hpp>
#include <legate_dataframe/unaryop.hpp>

int main(int argc, char** argv)
{
  // First we initialize Legate use either `legate` or `LEGATE_CONFIG` to customize launch
  legate::start();

  // Then let's create a new logical column
  legate::dataframe::LogicalColumn col_a = legate::dataframe::sequence(10, 0);

  // Compute the absolute value of each row in `col_a`
  legate::dataframe::LogicalColumn col_b = unary_operation(col_a, cudf::unary_operator::ABS);

  // And print
  std::cout << col_b.repr() << std::endl;
  return 0;
}
