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

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/table/table_device_view.cuh>

#include <arrow/compute/api.h>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

template <typename T, typename Enable = void>
struct copy_into_transposed_fn {
  template <typename... Args>
  void operator()(Args&&...)
  {
    throw std::runtime_error("copy_into_transposed(): type not supported");
  }
};

struct copy_into_transposed_impl {
  template <typename T>
  void operator()(TaskContext& ctx,
                  legate::PhysicalArray& array,
                  cudf::table_view tbl,
                  size_t offset,
                  legate::Scalar& null_value)
  {
    copy_into_transposed_fn<T>{}(ctx, array, tbl, offset, null_value);
  }

  template <typename T>
  void operator()(TaskContext& ctx,
                  void* data_ptr,
                  std::optional<bool*> null_ptr,
                  cudf::table_view tbl,
                  legate::Scalar& null_value)
  {
    copy_into_transposed_fn<T>{}(ctx, data_ptr, null_ptr, tbl, null_value);
  }
};

template <typename T>
struct copy_into_transposed_fn<T, std::enable_if_t<cudf::is_rep_layout_compatible<T>()>> {
  void operator()(TaskContext& ctx,
                  legate::PhysicalArray& array,
                  cudf::table_view tbl,
                  size_t offset,
                  legate::Scalar& null_value)
  {
    legate::Rect<2> bounds{{offset, 0}, {offset + tbl.num_rows() - 1, tbl.num_columns() - 1}};
    if (bounds.empty()) { return; }

    auto acc = array.data().write_accessor<T, 2, true>();
    if (!acc.accessor.is_dense_row_major(bounds)) {
      throw std::runtime_error("internal error: copy_into_transpose assume C-order store (data).");
    }
    T* data_ptr = acc.ptr(bounds.lo);
    std::optional<bool*> null_ptr{};

    if (array.nullable()) {
      auto mask_acc = array.null_mask().write_accessor<bool, 2, true>();
      if (!mask_acc.accessor.is_dense_row_major(bounds)) {
        throw std::runtime_error(
          "internal error: copy_into_transpose assume C-order store (mask).");
      }
      null_ptr = mask_acc.ptr(bounds.lo);
    }

    copy_into_transposed_fn<T>{}(ctx, data_ptr, null_ptr, tbl, null_value);
  }

  void operator()(TaskContext& ctx,
                  void* data_ptr_void,
                  std::optional<bool*> null_ptr,
                  cudf::table_view tbl,
                  legate::Scalar& null_value)
  {
    T* data_ptr = static_cast<T*>(data_ptr_void);
    // Similar to cudf's interleave_columns (we don't want to allocate, so avoid it).
    auto device_input = cudf::table_device_view::create(tbl, ctx.stream());

    auto index_begin = thrust::make_counting_iterator<size_t>(0);
    auto index_end   = thrust::make_counting_iterator<size_t>(tbl.num_rows() * tbl.num_columns());

    if (!null_ptr.has_value()) {
      // Our null value may be empty if the user didn't specify one (e.g. when there are no nulls).
      // Accessing the empty scalar would then cause an exception.
      T scalar{};
      if (null_value.size() > 0) { scalar = null_value.value<T>(); }
      auto get_value_func = cuda::proclaim_return_type<T>(
        [input = *device_input, divisor = tbl.num_columns(), scalar] __device__(size_t idx) {
          if (input.column(idx % divisor).is_valid(idx / divisor)) {
            return input.column(idx % divisor).element<T>(idx / divisor);
          } else {
            return scalar;
          }
        });

      thrust::transform(
        rmm::exec_policy(ctx.stream()), index_begin, index_end, data_ptr, get_value_func);
    } else {
      // This assumes that for rep_layout_compatible types `.element<T>(idx)` is OK even for masked
      // values.
      auto get_value_func = cuda::proclaim_return_type<T>(
        [input = *device_input, divisor = tbl.num_columns()] __device__(size_t idx) {
          return input.column(idx % divisor).element<T>(idx / divisor);
        });

      thrust::transform(
        rmm::exec_policy(ctx.stream()), index_begin, index_end, data_ptr, get_value_func);

      auto get_isvalid_func = cuda::proclaim_return_type<bool>(
        [input = *device_input, divisor = tbl.num_columns()] __device__(size_t idx) {
          return input.column(idx % divisor).is_valid_nocheck(idx / divisor);
        });

      thrust::transform(
        rmm::exec_policy(ctx.stream()), index_begin, index_end, null_ptr.value(), get_isvalid_func);
    }
  }
};

}  // namespace

void copy_into_tranposed(TaskContext& ctx,
                         legate::PhysicalArray& array,
                         cudf::table_view tbl,
                         size_t offset,
                         legate::Scalar& null_value)
{
  for (auto&& col : tbl) {
    if (to_cudf_type_id(array.type().code()) != col.type().id()) {
      throw std::runtime_error("internal error: column types changed between files?");
    }
  }

  cudf::type_dispatcher(
    tbl.column(0).type(), copy_into_transposed_impl{}, ctx, array, tbl, offset, null_value);
}

void copy_into_tranposed(TaskContext& ctx,
                         void* data_ptr,
                         std::optional<bool*> null_ptr,
                         cudf::table_view tbl,
                         legate::Scalar& null_value)
{
  cudf::type_dispatcher(
    tbl.column(0).type(), copy_into_transposed_impl{}, ctx, data_ptr, null_ptr, tbl, null_value);
}

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
