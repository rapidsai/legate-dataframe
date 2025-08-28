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

#include <legate_dataframe/core/column.hpp>

#include <arrow/compute/api.h>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

struct TransposeVisitor {
  void* data_ptr;
  std::optional<bool*> null_ptr;
  legate::Scalar& null_value;
  int column_idx;
  std::size_t num_columns;
  std::size_t row_offset;
  template <typename Type>
  arrow::Status Visit(const arrow::NumericArray<Type>& array)
  {
    using T         = typename std::decay_t<decltype(array)>::TypeClass::c_type;
    auto array_data = array.raw_values();
    auto out        = static_cast<T*>(data_ptr);
    if (!null_ptr.has_value()) {
      for (auto row_idx = row_offset; row_idx < row_offset + array.length(); row_idx++) {
        out[num_columns * row_idx + column_idx] = array.IsValid(row_idx - row_offset)
                                                    ? array_data[row_idx - row_offset]
                                                    : null_value.value<T>();
      }
    } else {
      auto null_data = null_ptr.value();
      for (auto row_idx = row_offset; row_idx < row_offset + array.length(); row_idx++) {
        null_data[num_columns * row_idx + column_idx] = array.IsValid(row_idx - row_offset);
        out[num_columns * row_idx + column_idx]       = array_data[row_idx - row_offset];
      }
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const arrow::BooleanArray& array)
  {
    auto out = static_cast<bool*>(data_ptr);
    if (!null_ptr.has_value()) {
      for (auto row_idx = row_offset; row_idx < row_offset + array.length(); row_idx++) {
        out[num_columns * row_idx + column_idx] = array.IsValid(row_idx - row_offset)
                                                    ? array.Value(row_idx - row_offset)
                                                    : null_value.value<bool>();
      }
    } else {
      auto null_data = null_ptr.value();
      for (auto row_idx = row_offset; row_idx < row_offset + array.length(); row_idx++) {
        null_data[num_columns * row_idx + column_idx] = array.IsValid(row_idx - row_offset);
        out[num_columns * row_idx + column_idx]       = array.Value(row_idx - row_offset);
      }
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const arrow::Array& array)
  {
    return arrow::Status::NotImplemented("Not implemented for array of type ",
                                         array.type()->ToString());
  }
};

void copy_into_tranposed(TaskContext& ctx,
                         void* data_ptr,
                         std::optional<bool*> null_ptr,
                         std::shared_ptr<arrow::Table> table,
                         legate::Scalar& null_value,
                         legate::Type type)
{
  // Iterate over columns and copy them into the data_ptr.
  // If the array is nullable, replace with value
  for (int i = 0; i < table->num_columns(); i++) {
    auto chunked_array     = table->column(i);
    std::size_t row_offset = 0;
    for (int chunk = 0; chunk < chunked_array->num_chunks(); chunk++) {
      auto array = chunked_array->chunk(chunk);
      TransposeVisitor visitor{.data_ptr    = data_ptr,
                               .null_ptr    = null_ptr,
                               .null_value  = null_value,
                               .column_idx  = i,
                               .num_columns = static_cast<std::size_t>(table->num_columns()),
                               .row_offset  = row_offset};

      // Cast if necessary
      auto target_arrow_type = to_arrow_type(to_cudf_type_id(type.code()));
      if (array->type_id() != target_arrow_type->id()) {
        auto casted_array = ARROW_RESULT(arrow::compute::Cast(*array, target_arrow_type));
        array             = std::move(casted_array);
      }

      auto status = arrow::VisitArrayInline(*array, &visitor);
      if (!status.ok()) {
        throw std::invalid_argument("from_arrow(): failed to transpose arrow array: " +
                                    status.ToString());
      }
      row_offset += array->length();
    }
  }
}

}  // namespace legate::dataframe
