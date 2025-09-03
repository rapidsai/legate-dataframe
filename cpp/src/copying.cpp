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

#include <arrow/compute/api.h>
#include <legate_dataframe/copying.hpp>
#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/library.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

/*static*/ void CopyIfElseTask::cpu_variant(legate::TaskContext context)
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

  std::vector<arrow::Datum> args;
  args.reserve(3);
  for (auto& column : {&cond, &lhs, &rhs}) {
    if (column->num_rows() == 1) {
      auto scalar = ARROW_RESULT(column->arrow_array_view()->GetScalar(0));
      args.emplace_back(scalar);
    } else {
      args.emplace_back(column->arrow_array_view());
    }
  }

  // Result may be scalar or array
  auto datum_result = ARROW_RESULT(arrow::compute::CallFunction("if_else", args));

  if (datum_result.is_scalar()) {
    auto as_array = ARROW_RESULT(arrow::MakeArrayFromScalar(*datum_result.scalar(), 1));
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(as_array));
    } else {
      output.move_into(std::move(as_array));
    }
  } else {
    if (get_prefer_eager_allocations()) {
      output.copy_into(std::move(datum_result.make_array()));
    } else {
      output.move_into(std::move(datum_result.make_array()));
    }
  }
}

struct copy_store_fn {
  template <legate::Type::Code CODE>
  void operator()(const legate::PhysicalStore& input, legate::PhysicalStore& output)
  {
    using value_type = legate::type_of_t<CODE>;
    std::array<size_t, 1> in_strides{};
    std::array<size_t, 1> out_strides{};
    auto in_ptr = input.read_accessor<value_type, 1>().ptr(input.shape<1>(), in_strides.data());
    auto out_ptr =
      output.write_accessor<value_type, 1>().ptr(output.shape<1>(), out_strides.data());
    assert(input.shape<1>().volume() == output.shape<1>().volume());
    assert(input.shape<1>().volume() <= 1 || (in_strides[0] == 1 && out_strides[0] == 1));
    memcpy(out_ptr, in_ptr, input.shape<1>().volume() * sizeof(value_type));
  }
};

/*static*/ void CopyTask::cpu_variant(legate::TaskContext context)
{
  TaskContext ctx{context};
  const auto input = ctx.get_next_input_arg();
  auto output      = ctx.get_next_output_arg();

  if (input.shape<1>().volume() <= 0) {
    return;  // Nothing to do, but e.g. pointer getting might fail.
  }

  auto in_store  = input.data();
  auto out_store = output.data();
  legate::type_dispatch(input.type().code(), copy_store_fn{}, in_store, out_store);
  if (input.nullable()) {
    auto null_mask_in_store  = input.null_mask();
    auto null_mask_out_store = output.null_mask();
    copy_store_fn{}.operator()<legate::Type::Code::BOOL>(null_mask_in_store, null_mask_out_store);
  } else if (output.nullable()) {
    auto out_acc = output.null_mask().write_accessor<bool, 1>();
    memset(out_acc.ptr(output.shape<1>()), true, output.shape<1>().volume() * sizeof(bool));
  }
}

/*static*/ void CopyOffsetsTask::cpu_variant(legate::TaskContext context)
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
  for (size_t i = 0; i < input.shape<1>().volume(); ++i) {
    out_ptr[i].lo[0] = in_ptr[i].lo[0] + offset;
    out_ptr[i].hi[0] = in_ptr[i].hi[0] + offset;
  }

  if (input.nullable()) {
    auto null_mask_in_store  = input.null_mask();
    auto null_mask_out_store = output.null_mask();
    copy_store_fn{}.operator()<legate::Type::Code::BOOL>(null_mask_in_store, null_mask_out_store);
  } else if (output.nullable()) {
    auto out_acc = output.null_mask().write_accessor<bool, 1>();
    memset(out_acc.ptr(output.shape<1>()), true, output.shape<1>().volume() * sizeof(bool));
  }
}

}  // namespace legate::dataframe::task

namespace {

void __attribute__((constructor)) register_tasks()
{
  legate::dataframe::task::CopyIfElseTask::register_variants();
  legate::dataframe::task::CopyTask::register_variants();
  legate::dataframe::task::CopyOffsetsTask::register_variants();
}

}  // namespace

namespace legate::dataframe {

LogicalColumn copy_if_else(const LogicalColumn& cond,
                           const LogicalColumn& lhs,
                           const LogicalColumn& rhs)
{
  if (*lhs.arrow_type() != *rhs.arrow_type()) {
    throw std::invalid_argument("lhs and rhs must have the same type");
  }
  if (*cond.arrow_type() != *arrow::boolean()) {
    throw std::invalid_argument("cond must be a boolean column");
  }

  auto runtime = legate::Runtime::get_runtime();

  bool nullable      = lhs.nullable() || rhs.nullable();
  auto scalar_result = lhs.is_scalar() && rhs.is_scalar() && cond.is_scalar();
  std::optional<size_t> size{};
  if (get_prefer_eager_allocations()) {
    size = std::max(lhs.num_rows(), std::max(rhs.num_rows(), cond.num_rows()));
  }
  auto ret = LogicalColumn::empty_like(lhs.arrow_type(), nullable, scalar_result, size);
  legate::AutoTask task =
    runtime->create_task(get_library(), task::CopyIfElseTask::TASK_CONFIG.task_id());

  /* Add the inputs, broadcast if scalar.  If both aren't scalar align them */
  auto cond_var = argument::add_next_input(task, cond, /* broadcast */ cond.is_scalar());
  auto lhs_var  = argument::add_next_input(task, lhs, /* broadcast */ lhs.is_scalar());
  auto rhs_var  = argument::add_next_input(task, rhs, /* broadcast */ rhs.is_scalar());
  if (!lhs.is_scalar()) { task.add_constraint(legate::align(lhs_var, cond_var)); }
  if (!rhs.is_scalar()) { task.add_constraint(legate::align(rhs_var, cond_var)); }
  auto out_var = argument::add_next_output(task, ret);
  if (size.has_value()) { task.add_constraint(legate::align(out_var, cond_var)); }
  runtime->submit(std::move(task));
  return ret;
}

LogicalColumn concatenate(const std::vector<LogicalColumn>& columns)
{
  if (columns.empty()) { throw std::invalid_argument("columns must not be empty"); }
  auto runtime = legate::Runtime::get_runtime();

  size_t size   = 0;
  bool nullable = false;
  for (auto& column : columns) {
    if (*column.arrow_type() != *columns[0].arrow_type()) {
      throw std::invalid_argument("columns must have the same type");
    }
    if (column.nullable()) { nullable = true; }
    size += column.num_rows();
  }

  if (columns[0].type().code() != legate::Type::Code::STRING) {
    auto res = LogicalColumn::empty_like(columns[0].arrow_type(), nullable, false, size);

    int64_t offset = 0;
    // We use a copy task for each slice (hopefully/should deal with string offsets correctly)
    // TODO(seberg): `issue_copy()` doesn't work (yet?) because the slice is a transformed store.
    for (auto& column : columns) {
      auto task   = runtime->create_task(get_library(), task::CopyTask::TASK_CONFIG.task_id());
      auto in_var = task.add_input(column.get_logical_array());
      auto out_var =
        task.add_output(res.get_logical_array().slice(0, {offset, offset + column.num_rows()}));

      task.add_constraint(legate::align(out_var, in_var));
      runtime->submit(std::move(task));
      offset += column.num_rows();
    }
    return res;
  }

  auto out_string_offsets = runtime->create_array({size}, legate::rect_type(1), nullable);

  /* Dealing with strings and offsets is a lot more complex unfortunately. */
  size_t row_offset = 0;
  size_t num_chars  = 0;
  for (auto& column : columns) {
    auto string_arr = column.get_logical_array().as_string_array();

    // Copy offsets (and null mask) adjusting for the actual final offsets in the result array./
    auto task   = runtime->create_task(get_library(), task::CopyOffsetsTask::TASK_CONFIG.task_id());
    auto in_var = task.add_input(string_arr.offsets());
    auto out_var =
      task.add_output(out_string_offsets.slice(0, {row_offset, row_offset + column.num_rows()}));
    task.add_scalar_arg(num_chars);
    task.add_constraint(legate::align(out_var, in_var));
    runtime->submit(std::move(task));

    row_offset += column.num_rows();
    // count how many characters we have until now (and in total).
    num_chars += string_arr.chars().data().shape().volume();
  }

  // create result char buffer:
  auto out_char_array = runtime->create_array({num_chars}, legate::int8(), false);

  num_chars = 0;
  for (auto& column : columns) {
    auto string_arr = column.get_logical_array().as_string_array();
    auto curr_chars = string_arr.chars().data().shape().volume();

    auto task    = runtime->create_task(get_library(), task::CopyTask::TASK_CONFIG.task_id());
    auto in_var  = task.add_input(string_arr.chars());
    auto out_var = task.add_output(out_char_array.slice(0, {num_chars, num_chars + curr_chars}));
    task.add_constraint(legate::align(out_var, in_var));
    runtime->submit(std::move(task));

    num_chars += curr_chars;
  }

  return runtime->create_string_array(out_string_offsets, out_char_array);
}

}  // namespace legate::dataframe
