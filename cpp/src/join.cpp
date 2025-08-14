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

#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <legate.h>

#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/join.hpp>

namespace legate::dataframe {
namespace task {
bool is_repartition_not_needed(const TaskContext& ctx,
                               JoinType join_type,
                               bool lhs_broadcasted,
                               bool rhs_broadcasted)
{
  if (ctx.nranks == 1) {
    return true;
  } else if (join_type == JoinType::INNER && (lhs_broadcasted || rhs_broadcasted)) {
    return true;
  } else if (join_type == JoinType::LEFT && rhs_broadcasted) {
    return true;
  } else {
    if (ctx.get_legate_context().communicators().size() == 0) {
      throw std::runtime_error(
        "internal join error: repartitioning needed but communicator not set up.");
    }
    return false;
  }
}

namespace {

/**
 * @brief Help function to map legate-dataframe join type to Arrow join type
 */
arrow::acero::JoinType legate_to_arrow_join_type(JoinType join_type)
{
  static const std::map<JoinType, arrow::acero::JoinType> join_type_map = {
    {JoinType::INNER, arrow::acero::JoinType::INNER},
    {JoinType::LEFT, arrow::acero::JoinType::LEFT_OUTER},
    {JoinType::FULL, arrow::acero::JoinType::FULL_OUTER}};

  auto it = join_type_map.find(join_type);
  if (it == join_type_map.end()) {
    throw std::invalid_argument("Unsupported join type for Arrow conversion");
  }
  return it->second;
}

std::shared_ptr<arrow::Table> revert_broadcast_arrow(TaskContext& ctx, const PhysicalTable& table)
{
  auto arrow_table = table.arrow_table_view();
  if (ctx.rank == 0 || table.is_partitioned()) {
    return arrow_table;
  } else {
    return ARROW_RESULT(arrow::Table::MakeEmpty(arrow_table->schema()));
  }
}

std::vector<std::string> integer_to_string_vector(const std::vector<int32_t>& vec)
{
  std::vector<std::string> result;
  result.reserve(vec.size());
  for (const auto& v : vec) {
    result.push_back(std::to_string(v));
  }
  return result;
}

}  // namespace

// Creating some constant for each arrow type is surprisingly hard
std::shared_ptr<arrow::Scalar> create_default_value_scalar(
  const std::shared_ptr<arrow::DataType>& type)
{
  // Constructing a string/binary type with 0 is a runtime error
  // Catch this and try constructing with a string
  try {
    return ARROW_RESULT(arrow::MakeScalar(type, 0));
  } catch (const std::exception& e) {
    return ARROW_RESULT(arrow::MakeScalar(type, std::string{}));
  }
}

/*
 * Helper function to convert a key column to not contain nulls and instead
 * add another explicit column that signals where the nulls are.
 *
 * May mutate fields and return a new table if the key column is nullable.
 */
std::shared_ptr<arrow::Table> process_key_column(std::shared_ptr<arrow::Table>& table,
                                                 const std::string& key,
                                                 std::shared_ptr<arrow::ChunkedArray>& key_array,
                                                 std::vector<arrow::FieldRef>& fields,
                                                 bool has_nulls,
                                                 bool needs_null_masks)
{
  if (!has_nulls) {
    fields.emplace_back(arrow::FieldRef(key));
  } else {
    auto scalar = create_default_value_scalar(key_array->type());
    auto key_replaced_nulls =
      ARROW_RESULT(arrow::compute::CallFunction("coalesce", {key_array, scalar})).chunked_array();

    table =
      ARROW_RESULT(table->AddColumn(table->num_columns(),
                                    arrow::field(key + "_replaced", key_replaced_nulls->type()),
                                    key_replaced_nulls));

    fields.emplace_back(arrow::FieldRef(key + "_replaced"));
  }

  if (!needs_null_masks) { return table; }

  arrow::ArrayVector null_chunks;
  for (auto chunk : key_array->chunks()) {
    std::shared_ptr<arrow::Array> null_chunk_array;
    if (chunk->null_bitmap()) {
      auto null_buffer = chunk->null_bitmap();
      null_chunk_array = arrow::MakeArray(
        arrow::ArrayData::Make(arrow::boolean(), chunk->length(), {nullptr, null_buffer}));
    } else {
      auto false_scalar = ARROW_RESULT(arrow::MakeScalar(arrow::boolean(), true));
      null_chunk_array  = ARROW_RESULT(arrow::MakeArrayFromScalar(*false_scalar, chunk->length()));
    }
    null_chunks.push_back(null_chunk_array);
  }

  auto null_chunked_array = std::make_shared<arrow::ChunkedArray>(null_chunks, arrow::boolean());
  table                   = ARROW_RESULT(table->AddColumn(
    table->num_columns(), arrow::field(key + "_mask", arrow::boolean()), null_chunked_array));

  fields.emplace_back(arrow::FieldRef(key + "_mask"));

  return table;
}

// Here we have to do some hacking because arrow does not support nulls in joins
// The general approach is to take the original key and demote it to a value
// Then take a copy of this demoted column, replace the nulls with a sentinel value
// Perform the join and then remove the extra columns
std::shared_ptr<arrow::Table> arrow_join_and_gather(TaskContext& ctx,
                                                    std::shared_ptr<arrow::Table> lhs,
                                                    std::shared_ptr<arrow::Table> rhs,
                                                    const std::vector<std::string> lhs_keys,
                                                    const std::vector<std::string> rhs_keys,
                                                    JoinType join_type,
                                                    bool nulls_are_equal,
                                                    const std::vector<std::string> lhs_out_cols,
                                                    const std::vector<std::string> rhs_out_cols)
{
  auto lhs_temp = lhs->Slice(0);
  auto rhs_temp = rhs->Slice(0);

  std::vector<arrow::FieldRef> left_fields;
  std::vector<arrow::FieldRef> right_fields;

  // If `nulls_are_equal` is not used, we can use the standard arrow join, but arrow
  // does not support `nulls_are_equal` so there we remove null and add null columns.
  if (!nulls_are_equal) {
    left_fields  = std::vector<arrow::FieldRef>(lhs_keys.begin(), lhs_keys.end());
    right_fields = std::vector<arrow::FieldRef>(rhs_keys.begin(), rhs_keys.end());
  } else {
    for (size_t i = 0; i < lhs_keys.size(); ++i) {
      // Process left key
      const auto& lhs_key = lhs_keys[i];
      auto lhs_key_array  = lhs->GetColumnByName(lhs_key);

      // Process right key
      const auto& rhs_key = rhs_keys[i];
      auto rhs_key_array  = rhs->GetColumnByName(rhs_key);

      auto lhs_has_nulls    = lhs_key_array->null_count();
      auto rhs_has_nulls    = rhs_key_array->null_count();
      bool needs_null_masks = lhs_has_nulls || rhs_has_nulls;

      // If needed replace original key column with a non-nulled one and a mask column
      lhs_temp = process_key_column(
        lhs_temp, lhs_key, lhs_key_array, left_fields, lhs_has_nulls, needs_null_masks);
      rhs_temp = process_key_column(
        rhs_temp, rhs_key, rhs_key_array, right_fields, rhs_has_nulls, needs_null_masks);
    }
  }

  std::vector<arrow::FieldRef> lhs_out_fields(lhs_out_cols.begin(), lhs_out_cols.end());
  std::vector<arrow::FieldRef> rhs_out_fields(rhs_out_cols.begin(), rhs_out_cols.end());

  arrow::acero::HashJoinNodeOptions join_opts(legate_to_arrow_join_type(join_type),
                                              left_fields,
                                              right_fields,
                                              lhs_out_fields,
                                              rhs_out_fields,
                                              arrow::compute::literal(true),
                                              "_left",
                                              "_right");

  arrow::acero::Declaration left{"table_source", arrow::acero::TableSourceNodeOptions{lhs_temp}};
  arrow::acero::Declaration right{"table_source", arrow::acero::TableSourceNodeOptions{rhs_temp}};
  arrow::acero::Declaration hashjoin{"hashjoin", {left, right}, std::move(join_opts)};

  return ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(hashjoin)));
}

template <bool needs_communication>
/*static*/ void JoinTask<needs_communication>::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto lhs          = argument::get_next_input<PhysicalTable>(ctx);
  const auto rhs          = argument::get_next_input<PhysicalTable>(ctx);
  const auto lhs_keys     = argument::get_next_scalar_vector<int32_t>(ctx);
  const auto rhs_keys     = argument::get_next_scalar_vector<int32_t>(ctx);
  auto join_type          = argument::get_next_scalar<JoinType>(ctx);
  auto null_equality      = argument::get_next_scalar<cudf::null_equality>(ctx);
  const auto lhs_out_cols = argument::get_next_scalar_vector<int32_t>(ctx);
  const auto rhs_out_cols = argument::get_next_scalar_vector<int32_t>(ctx);
  auto output             = argument::get_next_output<PhysicalTable>(ctx);

  /* Use "is_paritioned" to check if the table is broadcast. */
  const bool lhs_broadcasted = !lhs.is_partitioned();
  const bool rhs_broadcasted = !rhs.is_partitioned();
  if (lhs_broadcasted && rhs_broadcasted && ctx.nranks != 1) {
    throw std::runtime_error("join(): cannot have both the lhs and the rhs broadcasted");
  }

  auto arrow_lhs = lhs.arrow_table_view();
  auto arrow_rhs = rhs.arrow_table_view();

  std::shared_ptr<arrow::Table> result;
  if (is_repartition_not_needed(ctx, join_type, lhs_broadcasted, rhs_broadcasted)) {
    result = arrow_join_and_gather(ctx,

                                   arrow_lhs,
                                   arrow_rhs,
                                   integer_to_string_vector(lhs_keys),
                                   integer_to_string_vector(rhs_keys),
                                   join_type,
                                   null_equality == cudf::null_equality::EQUAL,
                                   integer_to_string_vector(lhs_out_cols),
                                   integer_to_string_vector(rhs_out_cols));
  } else {
    // All-to-all repartition to one hash bucket per rank. Matching rows from
    // both tables then guaranteed to be on the same rank.
    auto repartitioned_lhs = repartition_by_hash(ctx, revert_broadcast_arrow(ctx, lhs), lhs_keys);
    auto repartitioned_rhs = repartition_by_hash(ctx, revert_broadcast_arrow(ctx, rhs), rhs_keys);

    result = arrow_join_and_gather(ctx,
                                   repartitioned_lhs,
                                   repartitioned_rhs,
                                   integer_to_string_vector(lhs_keys),
                                   integer_to_string_vector(rhs_keys),
                                   join_type,
                                   null_equality == cudf::null_equality::EQUAL,
                                   integer_to_string_vector(lhs_out_cols),
                                   integer_to_string_vector(rhs_out_cols));
  }
  // Finally, create a vector of both the left and right results and move it into the output table
  if (get_prefer_eager_allocations() &&
      !output.unbound()) {  // hard to guess if bound so just inspect
    output.copy_into(std::move(result));
  } else {
    output.move_into(std::move(result));
  }
}
template class JoinTask<false>;
template class JoinTask<true>;
}  // namespace task

namespace {
/**
 * @brief Help function to append empty columns like those in `table`.
 */
void append_empty_like_columns(std::vector<LogicalColumn>& output,
                               const LogicalTable& table,
                               std::optional<size_t> size = std::nullopt)
{
  for (const auto& col : table.get_columns()) {
    output.push_back(
      LogicalColumn::empty_like(col.cudf_type(), col.nullable(), /* scalar */ false, size));
  }
}

/**
 * @brief Help function to append empty columns like those in `table`.
 * The empty columns are all nullable no matter the nullability of the columns in `table`
 */
void append_empty_like_columns_force_nullable(std::vector<LogicalColumn>& output,
                                              const LogicalTable& table,
                                              std::optional<size_t> size = std::nullopt)
{
  for (const auto& col : table.get_columns()) {
    output.push_back(LogicalColumn::empty_like(col.cudf_type(), true, /* scalar */ false, size));
  }
}
}  // namespace

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  const std::vector<size_t>& lhs_out_columns,
                  const std::vector<size_t>& rhs_out_columns,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  auto runtime = legate::Runtime::get_runtime();
  if (lhs_keys.size() != rhs_keys.size()) {
    throw std::invalid_argument("The size of `lhs_keys` and `rhs_keys` must be equal");
  }
  auto lhs_out = lhs.select(lhs_out_columns);
  auto rhs_out = rhs.select(rhs_out_columns);

  // Create an empty like table of the output columns
  std::vector<LogicalColumn> ret_cols;
  bool output_is_eager_lhs_aligned = false;
  switch (join_type) {
    case JoinType::INNER: {
      append_empty_like_columns(ret_cols, lhs_out);
      append_empty_like_columns(ret_cols, rhs_out);
      break;
    }
    case JoinType::LEFT: {
      if (broadcast == BroadcastInput::RIGHT && get_prefer_eager_allocations()) {
        output_is_eager_lhs_aligned = true;
        append_empty_like_columns(ret_cols, lhs_out, lhs.num_rows());
        append_empty_like_columns_force_nullable(ret_cols, rhs_out, lhs.num_rows());
      } else {
        append_empty_like_columns(ret_cols, lhs_out);
        append_empty_like_columns_force_nullable(ret_cols, rhs_out);
      }
      break;
    }
    case JoinType::FULL: {
      // In a full join, both left and right columns might contain nulls
      // even when `lhs` or `rhs` doesn't
      append_empty_like_columns_force_nullable(ret_cols, lhs_out);
      append_empty_like_columns_force_nullable(ret_cols, rhs_out);
      break;
    }
    default: {
      throw std::invalid_argument("Unknown JoinType");
    }
  }

  // Some calls broadcast inputs and thus do not need communication, in that case we
  // can use the non-concurrent task variant and do not need to add a communicator.
  bool needs_communication = false;
  if (broadcast == BroadcastInput::AUTO) {
    needs_communication = true;
  } else if (join_type == JoinType::FULL ||
             (broadcast == BroadcastInput::LEFT && join_type != JoinType::INNER)) {
    throw std::runtime_error(
      "Force broadcast was indicated, but repartitioning is required. "
      "FULL joins do not support broadcasting and LEFT joins only for the "
      "right hand side argument.");
  }

  // Create the output table
  auto ret_names = concat(lhs_out.get_column_name_vector(), rhs_out.get_column_name_vector());
  auto ret       = LogicalTable(std::move(ret_cols), std::move(ret_names));

  auto task_id =
    (needs_communication ? legate::dataframe::task::JoinTask<true>::TASK_CONFIG.task_id()
                         : legate::dataframe::task::JoinTask<false>::TASK_CONFIG.task_id());
  legate::AutoTask task = runtime->create_task(get_library(), task_id);
  // TODO: While legate may broadcast some arrays, it would be good to add
  //       a heuristic (e.g. based on the fact that we need to do copies
  //       anyway, so the broadcast may actually copy less).
  //       That could be done here, in a mapper, or within the task itself.
  auto lhs_vars = argument::add_next_input(task, lhs, broadcast == BroadcastInput::LEFT);
  argument::add_next_input(task, rhs, broadcast == BroadcastInput::RIGHT);
  argument::add_next_scalar_vector(task, std::vector<int32_t>(lhs_keys.begin(), lhs_keys.end()));
  argument::add_next_scalar_vector(task, std::vector<int32_t>(rhs_keys.begin(), rhs_keys.end()));
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<JoinType>>(join_type));
  argument::add_next_scalar(
    task, static_cast<std::underlying_type_t<cudf::null_equality>>(compare_nulls));
  argument::add_next_scalar_vector(
    task, std::vector<int32_t>(lhs_out_columns.begin(), lhs_out_columns.end()));
  argument::add_next_scalar_vector(
    task, std::vector<int32_t>(rhs_out_columns.begin(), rhs_out_columns.end()));
  auto output_vars = argument::add_next_output(task, ret);

  if (output_is_eager_lhs_aligned) {
    for (auto& var : output_vars) {
      task.add_constraint(legate::align(var, lhs_vars.at(0)));
    }
  }

  if (needs_communication) {
    if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
      task.add_communicator("cpu");
    } else {
      task.add_communicator("nccl");
    }
  }
  runtime->submit(std::move(task));
  return ret;
}

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<std::string>& lhs_keys,
                  const std::set<std::string>& rhs_keys,
                  JoinType join_type,
                  const std::vector<std::string>& lhs_out_columns,
                  const std::vector<std::string>& rhs_out_columns,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  // Convert column names to indices
  std::set<size_t> lhs_keys_idx;
  std::set<size_t> rhs_keys_idx;
  std::vector<size_t> lhs_out_columns_idx;
  std::vector<size_t> rhs_out_columns_idx;
  const auto& lhs_name_to_idx = lhs.get_column_names();
  const auto& rhs_name_to_idx = rhs.get_column_names();
  for (const auto& name : lhs_keys) {
    lhs_keys_idx.insert(lhs_name_to_idx.at(name));
  }
  for (const auto& name : rhs_keys) {
    rhs_keys_idx.insert(rhs_name_to_idx.at(name));
  }
  for (const auto& name : lhs_out_columns) {
    lhs_out_columns_idx.push_back(lhs_name_to_idx.at(name));
  }
  for (const auto& name : rhs_out_columns) {
    rhs_out_columns_idx.push_back(rhs_name_to_idx.at(name));
  }
  return join(lhs,
              rhs,
              lhs_keys_idx,
              rhs_keys_idx,
              join_type,
              lhs_out_columns_idx,
              rhs_out_columns_idx,
              compare_nulls,
              broadcast);
}

LogicalTable join(const LogicalTable& lhs,
                  const LogicalTable& rhs,
                  const std::set<size_t>& lhs_keys,
                  const std::set<size_t>& rhs_keys,
                  JoinType join_type,
                  cudf::null_equality compare_nulls,
                  BroadcastInput broadcast)
{
  // By default, the output includes all the columns from `lhs` and `rhs`.
  std::vector<size_t> lhs_out_columns(lhs.num_columns());
  std::iota(lhs_out_columns.begin(), lhs_out_columns.end(), 0);
  std::vector<size_t> rhs_out_columns(rhs.num_columns());
  std::iota(rhs_out_columns.begin(), rhs_out_columns.end(), 0);
  return join(lhs,
              rhs,
              lhs_keys,
              rhs_keys,
              join_type,
              lhs_out_columns,
              rhs_out_columns,
              compare_nulls,
              broadcast);
}

}  // namespace legate::dataframe

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::JoinTask<true>::register_variants();
  legate::dataframe::task::JoinTask<false>::register_variants();
}

}  // namespace
