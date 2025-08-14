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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/reduction.hpp>

namespace legate::dataframe {
namespace task {

namespace {

std::unique_ptr<cudf::reduce_aggregation> make_cudf_reduce_aggregation(const std::string& agg_kind)
{
  if (agg_kind == "sum") {
    return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  } else if (agg_kind == "product") {
    return cudf::make_product_aggregation<cudf::reduce_aggregation>();
  } else if (agg_kind == "min") {
    return cudf::make_min_aggregation<cudf::reduce_aggregation>();
  } else if (agg_kind == "max") {
    return cudf::make_max_aggregation<cudf::reduce_aggregation>();
  } else if (agg_kind == "mean") {
    return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
  } else {
    throw std::invalid_argument("Unsupported aggregation kind: " + agg_kind);
  }
}
}  // namespace

void ReduceLocalTask::gpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};

  const auto input = argument::get_next_input<PhysicalColumn>(ctx);
  auto op          = argument::get_next_scalar<std::string>(ctx);
  auto finalize    = argument::get_next_scalar<bool>(ctx);
  auto initial     = argument::get_next_scalar<bool>(ctx);
  auto output      = argument::get_next_output<PhysicalColumn>(ctx);

  // Fetching initial value column below if used.

  auto col_view = input.column_view();
  std::unique_ptr<const cudf::scalar> scalar_res;
  // TODO: Counting is slightly awkward, it may be best if it was just
  // specially handled (once we have a count-valid function)
  if (op == "count_valid") {
    assert(!initial);
    if (!finalize) {
      auto count = col_view.size() - col_view.null_count();
      scalar_res =
        std::make_unique<cudf::scalar_type_t<int64_t>>(count, true, ctx.stream(), ctx.mr());
    } else {
      auto sum   = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto zero  = cudf::numeric_scalar<int64_t>(0, true, ctx.stream(), ctx.mr());
      scalar_res = cudf::reduce(col_view, *sum, output.cudf_type(), zero, ctx.stream(), ctx.mr());
    }
  } else {
    std::unique_ptr<rmm::device_buffer> new_mask;
    auto agg = make_cudf_reduce_aggregation(op);
    // As of 25.08.dev cudfs min/max don't guarantee consistent null handling.
    // Most code uses `nans_to_nulls` early on to deal with this.  We do it very late (i.e. here)
    // currently (also since pyarrow does deal with it).
    if ((op == "min" || op == "max") && cudf::is_floating_point(output.cudf_type())) {
      auto [new_mask_, new_null_count] = cudf::nans_to_nulls(col_view, ctx.stream(), ctx.mr());
      new_mask                         = std::move(new_mask_);
      assert(col_view.num_children() == 0);
      col_view = cudf::column_view(col_view.type(),
                                   col_view.size(),
                                   col_view.head<void>(),
                                   reinterpret_cast<cudf::bitmask_type*>(new_mask->data()),
                                   new_null_count,
                                   col_view.offset());
    }
    if (initial) {
      auto initial_col    = argument::get_next_input<PhysicalColumn>(ctx);
      auto initial_scalar = initial_col.cudf_scalar();
      scalar_res =
        cudf::reduce(col_view, *agg, output.cudf_type(), *initial_scalar, ctx.stream(), ctx.mr());
    } else {
      scalar_res = cudf::reduce(col_view, *agg, output.cudf_type(), ctx.stream(), ctx.mr());
    }
  }

  // Note: cudf has no helper to go to a column view right now, but we could
  // specialize this in principle.
  output.move_into(cudf::make_column_from_scalar(*scalar_res, 1, ctx.stream(), ctx.mr()));
}

}  // namespace task
}  // namespace legate::dataframe
