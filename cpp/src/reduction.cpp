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

#include <algorithm>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <legate.h>

#include <arrow/compute/api.h>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>  // cudf::detail::target_type
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/binaryop.hpp>
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

class ReduceLocalTask : public Task<ReduceLocalTask, OpCode::ReduceLocal> {
 public:
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{OpCode::ReduceLocal}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input = argument::get_next_input<PhysicalColumn>(ctx);
    auto op          = argument::get_next_scalar<std::string>(ctx);
    auto finalize    = argument::get_next_scalar<bool>(ctx);
    auto initial     = argument::get_next_scalar<bool>(ctx);
    auto output      = argument::get_next_output<PhysicalColumn>(ctx);

    auto array = input.arrow_array_view();
    std::shared_ptr<arrow::Array> result_array;
    if (op == "count_valid") {
      assert(!initial);
      if (!finalize) {
        auto count   = std::make_shared<arrow::Int64Scalar>(array->length() - array->null_count());
        result_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*count, 1));
      } else {
        auto sum     = ARROW_RESULT(arrow::compute::Sum(array));
        result_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*sum.scalar(), 1));
      }
    } else {
      if (initial) {
        auto initial_col     = argument::get_next_input<PhysicalColumn>(ctx);
        auto initial_array   = initial_col.arrow_array_view();
        auto result          = ARROW_RESULT(arrow::compute::CallFunction(op, {array})).scalar();
        auto result_as_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*result, 1));
        // Combine two arrays and reduce again
        auto combined = ARROW_RESULT(arrow::Concatenate({result_as_array, initial_array}));
        result        = ARROW_RESULT(arrow::compute::CallFunction(op, {combined})).scalar();
        result_array  = ARROW_RESULT(arrow::MakeArrayFromScalar(*result, 1));
      } else {
        auto result  = ARROW_RESULT(arrow::compute::CallFunction(op, {array}));
        result_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*result.scalar(), 1));
      }
    }

    // Cast if necessary
    auto expected_type = output.arrow_type();
    if (*result_array->type() != *expected_type) {
      result_array = ARROW_RESULT(arrow::compute::Cast(result_array, expected_type)).make_array();
    }
    output.move_into(result_array);
  }
  static void gpu_variant(legate::TaskContext context)
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
};

}  // namespace task

namespace {

/* Reductions that never need a nullable column */
const std::set<std::string> NEVER_NULL = {
  "count_valid",
  "any",
  "all",
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
 * @param output_type The desired result dtype.
 * @param initial The (user requested) initial result.  Only useful if finalizing.
 * @return A logical column containing the result.  If finalize is false this
 * has one entry per partition.  If true, will contain a single entry.
 */
LogicalColumn perform_simple_reduce(
  const LogicalColumn& col,
  std::string op,
  bool finalize,
  const cudf::data_type& output_type,
  std::optional<std::reference_wrapper<const LogicalColumn>> initial = std::nullopt)
{
  auto runtime = legate::Runtime::get_runtime();

  // with identity, result is never null (might be type dependent eventually).
  auto nullable = NEVER_NULL.count(op) == 0;
  if (nullable && initial.has_value()) {
    const LogicalColumn& initial_col = initial.value();
    nullable                         = initial_col.nullable();
  }
  auto ret = LogicalColumn::empty_like(output_type, nullable, /* scalar */ finalize);

  legate::AutoTask task =
    runtime->create_task(get_library(), task::ReduceLocalTask::TASK_CONFIG.task_id());

  // If we "finalize", gather all data to one worker via a broadcast constraint
  auto var = argument::add_next_input(task, col, /* broadcast */ finalize);
  argument::add_next_scalar(task, op);
  argument::add_next_scalar(task, finalize);
  argument::add_next_scalar(task, initial.has_value());
  argument::add_next_output(task, ret);
  if (initial.has_value()) {
    if (!finalize) {
      // If there is a use case in the future, simply remove this check.
      throw std::logic_error("initial doesn't make sense when not finalizing");
    }
    argument::add_next_input(task, initial.value(), true);
  }

  runtime->submit(std::move(task));
  return ret;
}

/*
 * Aggregations that can be implemented via `reduce -> gather -> reduce`, i.e.
 * they are associative.  `count` technically requires `count -> gather -> sum`
 * but we encode this on the task side (the second SUM also needs a zero initial).
 */
const std::set<std::string> SIMPLE_AGGS = {
  "any",
  "all",
  "min",
  "max",
  "product",
  "sum",
  "count_valid",
};

/*
 * Fully describe any aggregation and how to compare/hash them.
 * (If we use this for groupby, we would add the column name to this.)
 */
class AggregationDescriptor final {
 public:
  /*
   * Create an aggregation descriptor form an agg, output_type and optional
   * initial.  Note that the initial value is a pointer and thus not owned.
   * (As of now initial is always user provided here, so this is easy.)
   */
  AggregationDescriptor(std::string op,
                        cudf::data_type& output_type,
                        std::optional<std::reference_wrapper<const LogicalColumn>> initial)
    : op{op}, output_type{output_type}, initial_{initial} {};

  AggregationDescriptor(const AggregationDescriptor& other)
    : op{other.op}, output_type{other.output_type}, initial_{other.initial_} {};

  std::size_t do_hash() const
  {
    // Don't be fancy, so just xor the hash of the components
    return (std::hash<std::string>{}(op) ^ static_cast<std::size_t>(output_type.id()) ^
            output_type.scale() ^
            (initial_.has_value() ? reinterpret_cast<std::size_t>(&initial_.value().get()) : 0));
  };

  bool operator==(const AggregationDescriptor& other) const
  {
    // TODO(seberg): The dtype equality should be fine right now, but it is not
    // strictly correct for complicated dtypes (structs, lists).
    assert(output_type.id() != cudf::type_id::STRUCT && output_type.id() != cudf::type_id::LIST);
    return (op == other.op) && output_type.id() == other.output_type.id() &&
           output_type.scale() == other.output_type.scale() &&
           initial_.has_value() == other.initial_.has_value() &&
           (!initial_.has_value() || &initial_.value().get() == &other.initial_.value().get());
  };

  std::string op;
  const cudf::data_type output_type;

 private:
  std::optional<std::reference_wrapper<const LogicalColumn>> initial_;
};

struct hash_agg_descr {
  size_t operator()(const AggregationDescriptor& agg_descr) const { return agg_descr.do_hash(); }
};

/*
 * To do distributed aggregations we need to split them into first pass
 * aggregations that are calculated locally, but are not final and result
 * aggregations which make use of those results.
 * Additionally, some aggregations may be composites of others, such as mean
 * which is `sum/count`, while others like argmin/argmax would need a custom
 * finalization.
 *
 * (This helper is slightly overpowered, as it supports doing multiple
 * aggregations on the same column at once, however, that could be helpful.
 * It also designed in a way that it should be possible to specialize it for
 * other aggregations, such as group-by)
 */
class AggregationHelper final {
 public:
  AggregationHelper(const LogicalColumn& col) : col_{col} {}

  /*
   * Add a reduction request.  This will be broken down to individual reductions.
   *
   * WARNING: the `initial` value must outlive the `AggregationHelper` and its
   * pointer identity is used as a unique identifier.
   */
  void add(std::string op,
           cudf::data_type& output_type,
           std::optional<std::reference_wrapper<const LogicalColumn>> initial)
  {
    AggregationDescriptor agg_descr{op, output_type, initial};
    if (results_.count(agg_descr)) { return; }

    breakdown_aggregation(op, output_type);
    results_.try_emplace(agg_descr, std::nullopt);
  }

  /*
   * Helper to add a first pass request.  For now we assume that these never
   * need initial values, if they do we'll have to take care about ownership
   * (i.e. improve the way we compare the initial value).
   */
  void add_first_pass(std::string op, cudf::data_type output_type)
  {
    AggregationDescriptor agg_descr{op, output_type, std::nullopt};
    first_pass_results_.try_emplace(agg_descr, std::nullopt);
  }

  LogicalColumn get_result(std::string op,
                           cudf::data_type& output_type,
                           std::optional<std::reference_wrapper<const LogicalColumn>> initial)
  {
    AggregationDescriptor agg_descr{op, output_type, initial};

    auto res = results_.at(agg_descr);
    if (res.has_value()) { return res.value(); }

    LogicalColumn result = calculate_aggregation(op, output_type, initial);
    results_[agg_descr]  = result;
    return result;
  }

  LogicalColumn get_first_pass_result(std::string op, cudf::data_type& output_type)
  {
    AggregationDescriptor agg_descr{op, output_type, std::nullopt};
    return first_pass_results_.at(agg_descr).value();
  }

  /*
   * Note that the following functions are designed so that they can (hopefully)
   * be split out and re-used e.g. to improve groupby-aggregations.
   * (Doing this would require e.g. threading in column indices that would be
   * always 0 for the purposes here.)
   */

  /*
   * Function to find the breakdown aggregations needed to finalize this one.
   * That is, breakdown_aggregation is designed to be overloadable in theory
   * and uses `add` or `add_first_pass` to add any full or first path aggs
   * needed to do it's own finalization.
   */
  void breakdown_aggregation(std::string op, cudf::data_type& output_type)
  {
    if (op == "mean") {
      /* Mean calculates it's final result from the final sum and counts. */
      add("sum", output_type, std::nullopt);
      auto cudf_int64 = cudf::data_type{cudf::type_id::INT64};
      add("count_valid", cudf_int64, std::nullopt);
    } else {
      if (SIMPLE_AGGS.count(op) == 0) {
        throw std::invalid_argument("Aggregation kind is currently not supported: " + op);
      }
      add_first_pass(op, output_type);
    }
  }

  /*
   * Launch all first pass calculations (must be called before getting final results).
   *
   * Note: If we add argmax, it's task would also calculate the max and we should
   * prioritize it over a `max` launch here (i.e. there may be optimization left to do).
   */
  void do_first_pass()
  {
    for (auto& [agg_descr, col] : first_pass_results_) {
      col = perform_simple_reduce(col_, agg_descr.op, /* finalize */ false, agg_descr.output_type);
    }
  }

  /*
   * Function to calculate a final result (called exactly once for each agg).
   */
  LogicalColumn calculate_aggregation(
    std::string op,
    cudf::data_type& output_type,
    std::optional<std::reference_wrapper<const LogicalColumn>> initial)
  {
    if (op == "mean") {
      /* Mean calculates it's final result from the final sum and counts. */
      auto sum        = get_result("sum", output_type, std::nullopt);
      auto cudf_int64 = cudf::data_type{cudf::type_id::INT64};
      auto counts     = get_result("count_valid", cudf_int64, std::nullopt);
      return legate::dataframe::binary_operation(sum, counts, "divide", output_type);
    } else {
      auto first_pass_res = get_first_pass_result(op, output_type);
      return perform_simple_reduce(first_pass_res, op, /* finalize */ true, output_type, initial);
    }
  }

 private:
  /*
   * Results maps, we will fill these with std::nullopt to keep track of what
   * we still need to calculate.
   * (If we consider using this for groupby in the future, both result and
   * `AggregationDescriptor` may need an optional column name.)
   */
  const LogicalColumn col_;
  std::unordered_map<AggregationDescriptor, std::optional<LogicalColumn>, hash_agg_descr> results_;
  std::unordered_map<AggregationDescriptor, std::optional<LogicalColumn>, hash_agg_descr>
    first_pass_results_;
};

}  // namespace

LogicalColumn reduce(const LogicalColumn& col,
                     std::string reduce_op,
                     cudf::data_type output_type,
                     std::optional<std::reference_wrapper<const LogicalColumn>> initial)
{
  AggregationHelper agg_helper{col};
  agg_helper.add(reduce_op, output_type, initial);

  /* Aggregations are implemented in two passes, see above. */
  agg_helper.do_first_pass();
  return agg_helper.get_result(reduce_op, output_type, initial);
}

}  // namespace legate::dataframe

namespace {

const auto reg_id_ = []() -> char {
  legate::dataframe::task::ReduceLocalTask::register_variants();
  return 0;
}();

}  // namespace
