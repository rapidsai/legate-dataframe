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
};

template <typename T>
struct copy_into_transposed_fn<T, std::enable_if_t<cudf::is_rep_layout_compatible<T>()>> {
  void operator()(TaskContext& ctx,
                  legate::PhysicalArray& array,
                  cudf::table_view tbl,
                  size_t offset,
                  legate::Scalar& null_value)
  {
    auto stream = ctx.get_legate_context().get_task_stream();
    legate::Rect<2> bounds{{offset, 0}, {offset + tbl.num_rows() - 1, tbl.num_columns() - 1}};
    if (bounds.empty()) { return; }

    auto acc = array.data().write_accessor<T, 2, true>();
    if (!acc.accessor.is_dense_row_major(bounds)) {
      throw std::runtime_error("internal error: copy_into_transpose assume C-order store (data).");
    }

    // Similar to cudf's interleave_columns (we don't want to allocate, so avoid it).
    auto device_input = cudf::table_device_view::create(tbl, stream);

    auto index_begin = thrust::make_counting_iterator<size_t>(0);
    auto index_end   = thrust::make_counting_iterator<size_t>(bounds.volume());

    if (!array.nullable()) {
      auto get_value_func =
        cuda::proclaim_return_type<T>([input   = *device_input,
                                       divisor = tbl.num_columns(),
                                       val     = null_value.value<T>()] __device__(size_t idx) {
          if (input.column(idx % divisor).is_valid(idx / divisor)) {
            return input.column(idx % divisor).element<T>(idx / divisor);
          } else {
            return val;
          }
        });

      thrust::transform(
        rmm::exec_policy(stream), index_begin, index_end, acc.ptr(bounds.lo), get_value_func);
    } else {
      // This assumes that for rep_layout_compatible types `.element<T>(idx)` is OK even for masked
      // values.
      auto get_value_func = cuda::proclaim_return_type<T>(
        [input = *device_input, divisor = tbl.num_columns()] __device__(size_t idx) {
          return input.column(idx % divisor).element<T>(idx / divisor);
        });

      thrust::transform(
        rmm::exec_policy(stream), index_begin, index_end, acc.ptr(bounds.lo), get_value_func);

      auto get_isvalid_func = cuda::proclaim_return_type<bool>(
        [input = *device_input, divisor = tbl.num_columns()] __device__(size_t idx) {
          return input.column(idx % divisor).is_valid_nocheck(idx / divisor);
        });

      auto mask_acc = array.null_mask().write_accessor<bool, 2, true>();
      if (!mask_acc.accessor.is_dense_row_major(bounds)) {
        throw std::runtime_error(
          "internal error: copy_into_transpose assume C-order store (mask).");
      }

      thrust::transform(rmm::exec_policy(stream),
                        index_begin,
                        index_end,
                        mask_acc.ptr(bounds.lo),
                        get_isvalid_func);
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

}  // namespace legate::dataframe
