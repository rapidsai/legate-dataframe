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

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <legate.h>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/binaryop.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/reduction.hpp>

namespace legate::dataframe {
namespace task {

namespace {

/*
 * Helper just to get back from aggregation kind to actual aggregation object.
 */
std::unique_ptr<cudf::reduce_aggregation> make_reduce_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: {
      return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    }
    case cudf::aggregation::Kind::PRODUCT: {
      return cudf::make_product_aggregation<cudf::reduce_aggregation>();
    }
    case cudf::aggregation::Kind::MIN: {
      return cudf::make_min_aggregation<cudf::reduce_aggregation>();
    }
    case cudf::aggregation::Kind::MAX: {
      return cudf::make_max_aggregation<cudf::reduce_aggregation>();
    }
    case cudf::aggregation::Kind::SUM_OF_SQUARES: {
      return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
    }
    case cudf::aggregation::Kind::MEAN: {
      return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
    }
    default: {
      throw std::invalid_argument("Missing reduce aggregation mapping.");
    }
  }
}

}  // namespace

class ReduceLocalTask : public Task<ReduceLocalTask, OpCode::ReduceLocal> {
 public:
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    GPUTaskContext ctx{context};

    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto agg_kind    = argument::get_next_scalar<cudf::aggregation::Kind>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);

    /*
    if (agg_kind == cudf::aggregation::Kind::COUNT_VALID) {
      //  TODO: Make specialized reduce code, maybe along these lines?
      std::size_t null_count;
      if (!input.nullable()) {
        null_count = input.num_rows();
      }
      else {
        auto policy = DEFAULT_POLICY().on(ctx.stream());
        auto store  = input.get_byte_mask();
        auto shape  = store.shape<1>();
        bool *start = store.read_accessor<bool, 1, true>().ptr(shape.lo);
        bool *end   = start + (shape.hi[0] - shape.lo[0] + 1);

        null_count = thrust::count(policy, data, end, true)
      }
    }*/

    auto col_view = input.column_view();
    std::unique_ptr<const cudf::scalar> scalar_res;
    if (agg_kind == cudf::aggregation::Kind::COUNT_VALID) {
      // TODO: Null count should have a special implementation (this isn't ideal).
      auto count = col_view.size() - col_view.null_count();
      scalar_res =
        std::make_unique<cudf::scalar_type_t<int64_t>>(count, true, ctx.stream(), ctx.mr());
    } else {
      auto agg   = make_reduce_aggregation(agg_kind);
      scalar_res = cudf::reduce(col_view, *agg, output.cudf_type(), ctx.stream(), ctx.mr());
    }

    // Note: cudf has no helper to go to a column view right now, but we could
    // specialize this in principle.
    auto ret = cudf::make_column_from_scalar(*scalar_res, 1, ctx.stream(), ctx.mr());

    output.move_into(std::move(ret));
  }
};

}  // namespace task

namespace {

/* Reductions that have an identity (so their result is never null) */
const std::set<cudf::aggregation::Kind> has_identity = {
  cudf::aggregation::Kind::PRODUCT,
  cudf::aggregation::Kind::SUM,
};

/*
 * Perform a simple reduction.
 *
 * The caller must indicate whether this is a first-pass or second-pass
 * (finalizing) reduction.  And the caller must ensure that the aggregation
 * is sensible.
 *
 * @param col The column/values to reduce.
 * @param kind The aggregation kind (only supports simple aggs!).
 * @param finalize boolean to indicate which step we are in.  If false,
 * this is a first pass simple reduction, if true it is a finalizing one.
 *
 * @return A logical column containing the result.  If finalize is false this
 * has one entry per partition.  If true, will contain a single entry.
 */
LogicalColumn perform_simple_reduce(const LogicalColumn& col,
                                    cudf::aggregation::Kind kind,
                                    bool finalize)
{
  auto runtime = legate::Runtime::get_runtime();

  auto dtype = cudf::detail::target_type(col.cudf_type(), kind);
  // with identity, result is never null (might be type dependent eventually).
  auto nullable = has_identity.count(kind) == 0;
  auto ret = LogicalColumn::empty_like(dtype, nullable, /* scalar */ finalize);

  legate::AutoTask task = runtime->create_task(get_library(), task::ReduceLocalTask::TASK_ID);

  // If we "finalize", gather all data to one worker via a broadcast constraint
  auto var = argument::add_next_input(task, col, /* broadcast */ finalize);
  argument::add_next_scalar(task,
                            static_cast<std::underlying_type_t<cudf::aggregation::Kind>>(kind));
  argument::add_next_output(task, ret);

  runtime->submit(std::move(task));
  return ret;
}

/*
 * Aggregations that can be implemented via `reduce -> gather -> reduce`, i.e.
 * they are associative.  For example `count` is also trivial but requires
 * `count -> gather -> sum` (finalize being different).
 */
const std::set<cudf::aggregation::Kind> simple_aggs = {
  cudf::aggregation::Kind::MIN,
  cudf::aggregation::Kind::MAX,
  cudf::aggregation::Kind::PRODUCT,
  cudf::aggregation::Kind::SUM,
  // Sum of squares should likely be two pass (run after subtracting mean):
  // cudf::aggregation::Kind::SUM_OF_SQUARES,
};

/*
 * To do a distributed reduce, we have to always do local aggregations and then
 * reduce across all shards/workers.
 *
 * In general an aggregation such as `mean` cannot be computed locally, so the
 * "local aggregation" for a mean is to do both `sum` and `count` and then
 * finalize to `mean = sum/count`.
 *
 * We use the cudf internal simple-aggregations to decide which initial local
 * aggregations to do.  Then the finalizer visitor pattern to get final
 * reduce results (either from local results or from other final results).
 *
 * In general, we finalize local results by gathering all results (i.e.
 * broadcast) and doing a second local reduction.
 * In theory, that may not be ideal, e.g. a custom Legion reduction could
 * do this step with less communication, but the required data should be
 * small at this point.
 *
 * Not all aggregations can be described in this manner (e.g. median), but
 * missing ones that can certainly be:
 *   - argmin/argmax: Requires min+argmin and a specialized finalization.
 *   - std: Typically 2-passes (first to find mean), it's finalize would be
 *     complex (needs original data again) but probably works.
 */
class groupby_simple_aggregations_collector final
  : public cudf::detail::simple_aggregations_collector {
 public:
  using cudf::detail::simple_aggregations_collector::visit;

  std::vector<std::unique_ptr<cudf::aggregation>> visit(cudf::data_type col_type,
                                                        cudf::aggregation const& agg) override
  {
    if (simple_aggs.count(agg.kind) == 0) {
      throw std::invalid_argument("Aggregation kind is currently not supported.");
    }
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }

  std::vector<std::unique_ptr<cudf::aggregation>> visit(
    cudf::data_type col_type, cudf::detail::count_aggregation const& agg) override
  {
    /* Trivial aggregation, but split out as it needs a sum to finalize. */
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }

  std::vector<std::unique_ptr<cudf::aggregation>> visit(
    cudf::data_type col_type, cudf::detail::mean_aggregation const& agg) override
  {
    (void)col_type;
    CUDF_EXPECTS(cudf::is_fixed_width(col_type), "MEAN aggregation expects fixed width type");
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation());
    aggs.push_back(cudf::make_count_aggregation());  // COUNT_VALID
    return aggs;
  }
};

struct agg_ref_hash {
  size_t operator()(cudf::aggregation const& agg) const { return agg.do_hash(); }
};
struct agg_refs_equal {
  size_t operator()(cudf::aggregation const& lhs, cudf::aggregation const& rhs) const
  {
    return lhs.is_equal(rhs);
  }
};

/* Similar to libcudf result_cache pattern, but don't bother with dedicated class. */
class reduce_local_finalizer final : public cudf::detail::aggregation_finalizer {
 public:
  using agg_cache = std::unordered_map<
    std::reference_wrapper<cudf::aggregation const>,
    std::pair<legate::dataframe::LogicalColumn, std::unique_ptr<cudf::aggregation>>,
    agg_ref_hash,
    agg_refs_equal>;

  agg_cache first_pass_results;
  agg_cache final_results;

  reduce_local_finalizer(std::vector<std::unique_ptr<cudf::aggregation>>& aggs,
                         std::vector<legate::dataframe::LogicalColumn>& first_pass_results_)
  {
    for (size_t i = 0; i < aggs.size(); i++) {
      auto agg_copy   = aggs.at(i)->clone();
      auto const& key = *agg_copy;
      auto col        = first_pass_results_.at(i);
      first_pass_results.emplace(key, std::pair(std::move(col), std::move(agg_copy)));
    }
  }

  LogicalColumn get_final_result(cudf::aggregation const& agg)
  {
    agg.finalize(*this);
    return final_results.at(agg).first;
  }

  void set_final_result(cudf::aggregation const& agg, LogicalColumn& col)
  {
    auto agg_copy   = agg.clone();
    auto const& key = *agg_copy;
    final_results.emplace(key, std::pair(std::move(col), std::move(agg_copy)));
  }

  // Declare overloads for each kind of aggregation to dispatch
  void visit(cudf::aggregation const& agg) override
  {
    if (final_results.count(agg)) { return; }

    if (simple_aggs.count(agg.kind) == 0) {
      throw std::runtime_error("specific aggregation finalizer is missing: " +
                               std::to_string(agg.kind));
    }
    auto res =
      perform_simple_reduce(first_pass_results.at(agg).first, agg.kind, /* finalize */ true);
    set_final_result(agg, res);
    return;
  }

  void visit(cudf::detail::count_aggregation const& agg) override
  {
    if (final_results.count(agg)) { return; }

    auto res = perform_simple_reduce(
      first_pass_results.at(agg).first, cudf::aggregation::Kind::SUM, /* finalize */ true);
    set_final_result(agg, res);
  }

  void visit(cudf::detail::mean_aggregation const& agg) override
  {
    auto sum_agg    = cudf::make_sum_aggregation();
    auto counts_agg = cudf::make_count_aggregation();
    auto sum        = get_final_result(*sum_agg);
    auto counts     = get_final_result(*counts_agg);

    auto res_dtype = cudf::detail::target_type(sum.cudf_type(), cudf::aggregation::MEAN);
    auto res =
      legate::dataframe::binary_operation(sum, counts, cudf::binary_operator::DIV, res_dtype);
    set_final_result(agg, res);
  }
};

}  // namespace

// TODO: Need proper scalars for reduce result!
LogicalColumn reduce(const LogicalColumn& col, const cudf::reduce_aggregation& agg)
{
  /* Perform reduction in two steps (see comments above), start with simple aggs */
  groupby_simple_aggregations_collector preprocessor;
  auto initial_aggs = agg.get_simple_aggregations(col.cudf_type(), preprocessor);

  std::vector<legate::dataframe::LogicalColumn> first_pass_results;
  for (auto& initial_agg : initial_aggs) {
    first_pass_results.emplace_back(
      perform_simple_reduce(col, initial_agg->kind, /* finalize */ false));
  }

  reduce_local_finalizer finalizer(initial_aggs, first_pass_results);
  // Populate final result(s), needs to be done here for specific agg type dispatch.
  agg.finalize(finalizer);

  auto res = finalizer.get_final_result(agg);
  return res;
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::ReduceLocalTask::register_variants();
}

}  // namespace
