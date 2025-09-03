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

#include <legate.h>
#include <legate/cuda/cuda.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <cudf/copying.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/copying.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

/*static*/ void CopyIfElseTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto cond = argument::get_next_input<PhysicalColumn>(ctx);
  const auto lhs  = argument::get_next_input<PhysicalColumn>(ctx);
  const auto rhs  = argument::get_next_input<PhysicalColumn>(ctx);
  auto output     = argument::get_next_output<PhysicalColumn>(ctx);

  if (cond.num_rows() <= 0) {
    output.bind_empty_data();
    return;
  }

  std::unique_ptr<cudf::column> ret;
  /*
   * Use scalars if inputs are to ensure broadcasting works, cond is always a column.
   * This unfortunately requires 4 cases (all 4 overloads provided by libcudf).
   */
  if (lhs.num_rows() == 1 && rhs.num_rows() != 1) {
    auto lhs_scalar = lhs.cudf_scalar();
    ret             = cudf::copy_if_else(
      *lhs_scalar, rhs.column_view(), cond.column_view(), ctx.stream(), ctx.mr());
  } else if (rhs.num_rows() == 1 && lhs.num_rows() != 1) {
    auto rhs_scalar = rhs.cudf_scalar();
    ret             = cudf::copy_if_else(
      lhs.column_view(), *rhs_scalar, cond.column_view(), ctx.stream(), ctx.mr());
  } else if (lhs.num_rows() == 1 && rhs.num_rows() == 1) {
    auto lhs_scalar = lhs.cudf_scalar();
    auto rhs_scalar = rhs.cudf_scalar();
    ret = cudf::copy_if_else(*lhs_scalar, *rhs_scalar, cond.column_view(), ctx.stream(), ctx.mr());
  } else {
    ret = cudf::copy_if_else(
      lhs.column_view(), rhs.column_view(), cond.column_view(), ctx.stream(), ctx.mr());
  }

  if (get_prefer_eager_allocations()) {
    output.copy_into(std::move(ret));
  } else {
    output.move_into(std::move(ret));
  }
}

struct copy_store_fn {
  template <legate::Type::Code CODE>
  void operator()(TaskContext& ctx,
                  const legate::PhysicalStore& input,
                  legate::PhysicalStore& output)
  {
    using value_type = legate::type_of_t<CODE>;
    std::array<size_t, 1> in_strides{};
    std::array<size_t, 1> out_strides{};
    auto in_ptr = input.read_accessor<value_type, 1>().ptr(input.shape<1>(), in_strides.data());
    auto out_ptr =
      output.write_accessor<value_type, 1>().ptr(output.shape<1>(), out_strides.data());
    assert(input.shape<1>().volume() == output.shape<1>().volume());
    assert(input.shape<1>().volume() <= 1 || (in_strides[0] == 1 && out_strides[0] == 1));
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(out_ptr,
                                      in_ptr,
                                      input.shape<1>().volume() * sizeof(value_type),
                                      cudaMemcpyDeviceToDevice,
                                      ctx.stream()));
  }
};

/*static*/ void CopyTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = ctx.get_next_input_arg();
  auto output      = ctx.get_next_output_arg();

  if (input.shape<1>().volume() <= 0) {
    return;  // Nothing to do, but e.g. pointer getting might fail.
  }

  auto in_store  = input.data();
  auto out_store = output.data();
  legate::type_dispatch(input.type().code(), copy_store_fn{}, ctx, in_store, out_store);
  if (input.nullable()) {
    auto null_mask_in_store  = input.null_mask();
    auto null_mask_out_store = output.null_mask();
    copy_store_fn{}.operator()<legate::Type::Code::BOOL>(
      ctx, null_mask_in_store, null_mask_out_store);
  } else if (output.nullable()) {
    auto out_acc = output.null_mask().write_accessor<bool, 1>();
    LEGATE_CHECK_CUDA(cudaMemsetAsync(out_acc.ptr(output.shape<1>()),
                                      true,
                                      output.shape<1>().volume() * sizeof(bool),
                                      ctx.stream()));
  }
}

/*static*/ void CopyOffsetsTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = ctx.get_next_input_arg();
  auto output      = ctx.get_next_output_arg();
  auto offset      = argument::get_next_scalar<int64_t>(ctx);

  if (input.shape<1>().volume() <= 0) {
    return;  // Nothing to do, but e.g. pointer getting might fail.
  }

  auto in_store  = input.data();
  auto out_store = output.data();

  std::array<size_t, 1> in_strides{};
  std::array<size_t, 1> out_strides{};
  auto in_ptr =
    input.data().read_accessor<legate::Rect<1>, 1>().ptr(input.shape<1>(), in_strides.data());
  auto out_ptr =
    output.data().write_accessor<legate::Rect<1>, 1>().ptr(output.shape<1>(), out_strides.data());
  assert(input.shape<1>().volume() == output.shape<1>().volume());
  assert(input.shape<1>().volume() <= 1 || (in_strides[0] == 1 && out_strides[0] == 1));

  thrust::transform(thrust::cuda::par.on(ctx.stream()),
                    in_ptr,
                    in_ptr + input.shape<1>().volume(),
                    out_ptr,
                    [offset] __device__(const legate::Rect<1>& rect) {
                      return legate::Rect<1>{rect.lo[0] + offset, rect.hi[0] + offset};
                    });

  if (input.nullable()) {
    auto null_mask_in_store  = input.null_mask();
    auto null_mask_out_store = output.null_mask();
    copy_store_fn{}.operator()<legate::Type::Code::BOOL>(
      ctx, null_mask_in_store, null_mask_out_store);
  } else if (output.nullable()) {
    auto out_acc = output.null_mask().write_accessor<bool, 1>();
    LEGATE_CHECK_CUDA(cudaMemsetAsync(out_acc.ptr(output.shape<1>()),
                                      true,
                                      output.shape<1>().volume() * sizeof(bool),
                                      ctx.stream()));
  }
}

}  // namespace legate::dataframe::task
