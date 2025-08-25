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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <arrow/c/bridge.h> /* for arrow::ImportArray */

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/null_mask.hpp>
#include <legate_dataframe/core/print.hpp>
#include <legate_dataframe/core/ranges.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace detail {

struct ArrowToPhysicalArrayVisitor {
  ArrowToPhysicalArrayVisitor(legate::PhysicalArray& array) : array_(array) {}
  legate::PhysicalArray& array_;
  template <typename Type>
  arrow::Status Visit(const arrow::NumericArray<Type>& array)
  {
    using T = typename std::decay_t<decltype(array)>::TypeClass::c_type;
    if (sizeof(T) != array_.type().size()) {
      throw std::invalid_argument(
        "move_into(): the arrow column type size doesn't match the PhysicalArray");
    }
    T* out = maybe_bind_buffer<T>(array_.data(), array.length());
    std::memcpy(out, array.raw_values(), array.length() * sizeof(T));
    return arrow::Status::OK();
  }

  template <typename ArrayType,
            std::enable_if_t<std::is_same_v<ArrayType, arrow::StringArray> ||
                             std::is_same_v<ArrayType, arrow::LargeStringArray>>* = nullptr>
  arrow::Status Visit(const ArrayType& array)
  {
    auto legate_string_array = array_.as_string_array();
    auto ranges_size         = array.length();
    auto ranges =
      maybe_bind_buffer<legate::Rect<1>>(legate_string_array.ranges().data(), ranges_size);
    arrow_offsets_to_local_ranges(array, ranges);
    auto nbytes = array.total_values_length();
    auto chars  = maybe_bind_buffer<int8_t>(legate_string_array.chars().data(), nbytes);
    std::memcpy(chars, array.value_data()->data(), nbytes);
    return arrow::Status::OK();
  }
  arrow::Status Visit(const arrow::BooleanArray& array)
  {
    // Boolean array is bit packed
    auto out = maybe_bind_buffer<bool>(array_.data(), array.length());
    for (std::size_t i = 0; i < array.length(); ++i) {
      out[i] = array.Value(i);
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const arrow::Array& array)
  {
    return arrow::Status::NotImplemented("Not implemented for array of type ",
                                         array.type()->ToString());
  }
};

// Copy an arrow array into a physical array
// Binds the legate array if it is unbound
void from_arrow(legate::PhysicalArray array,
                std::shared_ptr<arrow::Array> arrow_array,
                bool scalar = false)
{
  if (array.type() != to_legate_type(*arrow_array->type())) {
    throw std::invalid_argument("from_arrow(): type mismatch: " + array.type().to_string() +
                                " != " + arrow_array->type()->ToString());
  }
  if (!array.nullable() && arrow_array->null_count() > 0) {
    throw std::invalid_argument("from_arrow(): arrow array has nulls but column is not nullable.");
  }
  if (scalar && arrow_array->length() != 1) {
    throw std::invalid_argument("from_arrow(): scalar column must have length 1.");
  }

  if (array.nullable()) {
    bool* null_mask;
    // If the array is a string, its null mask lives in a different place
    if (array.type().code() == legate::Type::Code::STRING) {
      null_mask =
        maybe_bind_buffer<bool>(array.as_string_array().null_mask(), arrow_array->length());
    } else {
      null_mask = maybe_bind_buffer<bool>(array.null_mask(), arrow_array->length());
    }

    for (size_t i = 0; i < arrow_array->length(); ++i) {
      null_mask[i] = arrow_array->IsValid(i);
    }
  }

  // Dispatch arrow::Array types
  ArrowToPhysicalArrayVisitor visitor{array};
  auto status = arrow::VisitArrayInline(*arrow_array, &visitor);
  if (!status.ok()) {
    throw std::invalid_argument("from_arrow(): failed to copy arrow array: " + status.ToString());
  }
}

// Copy an arrow array into a logical array
legate::LogicalArray from_arrow(std::shared_ptr<arrow::Array> arrow_array, bool scalar)
{
  // Create an unbound logical array
  auto arrow_has_nulls = arrow_array->null_count() > 0;
  auto runtime         = legate::Runtime::get_runtime();
  if (auto string_array = dynamic_cast<arrow::StringArray*>(arrow_array.get())) {
    auto array = runtime->create_string_array(
      runtime->create_array(
        {std::uint64_t(arrow_array->length())}, legate::rect_type(1), arrow_has_nulls),
      runtime->create_array({std::uint64_t(string_array->total_values_length())}, legate::int8()));
    from_arrow(array.get_physical_array(), arrow_array);
    return array;
  } else if (auto large_string_array = dynamic_cast<arrow::LargeStringArray*>(arrow_array.get())) {
    auto array = runtime->create_string_array(
      runtime->create_array(
        {std::uint64_t(arrow_array->length())}, legate::rect_type(1), arrow_has_nulls),
      runtime->create_array({std::uint64_t(large_string_array->total_values_length())},
                            legate::int8()));
    from_arrow(array.get_physical_array(), arrow_array);
    return array;
  }
  auto array = runtime->create_array({std::uint64_t(arrow_array->length())},
                                     to_legate_type(*arrow_array->type()),
                                     arrow_has_nulls,
                                     false /* scalar */);
  from_arrow(array.get_physical_array(), arrow_array);
  return array;
}

legate::LogicalArray from_arrow(std::shared_ptr<arrow::Scalar> scalar)
{
  auto array = ARROW_RESULT(arrow::MakeArrayFromScalar(*scalar, 1));
  return from_arrow(array);
}

};  // namespace detail

LogicalColumn::LogicalColumn(std::shared_ptr<arrow::Array> arrow_array)
  : LogicalColumn{// This type conversion monstrosity can be improved
                  detail::from_arrow(arrow_array),
                  to_cudf_type(arrow_array->type()),
                  /* scalar */ false}
{
}

LogicalColumn::LogicalColumn(std::shared_ptr<arrow::Scalar> arrow_scalar)
  : LogicalColumn{// This type conversion monstrosity can be improved
                  detail::from_arrow(arrow_scalar),
                  to_cudf_type(arrow_scalar->type),
                  /* scalar */ true}
{
}

std::shared_ptr<arrow::Array> LogicalColumn::get_arrow() const
{
  if (unbound()) {
    throw std::runtime_error(
      "Cannot call `.arrow_array()` on a unbound LogicalColumn, please bind it using "
      "`.move_into()`");
  }
  if (array_->nested()) {
    if (array_->type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_->get_physical_array().as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      const auto num_chars                = chars.data().shape<1>().volume();

      std::shared_ptr<arrow::Buffer> data =
        ARROW_RESULT(arrow::AllocateBuffer(num_chars * sizeof(int8_t)));
      std::memcpy(data->mutable_data(), read_accessor_as_1d_bytes(chars), num_chars);

      std::shared_ptr<arrow::Buffer> null_bitmask;
      if (a.nullable()) { null_bitmask = null_mask_bools_to_bits(a.null_mask()); }

      auto offsets = global_ranges_to_arrow_offsets(a.ranges().data());

      return std::make_shared<arrow::StringArray>(num_rows(), offsets, data, null_bitmask);

    } else {
      throw std::invalid_argument("nested dtype " + array_->type().to_string() +
                                  " isn't supported");
    }
  } else {
    auto physical_array = array_->get_physical_array();
    auto nbytes         = array_->volume() * array_->type().size();
    std::shared_ptr<arrow::Buffer> data;
    if (this->type().code() == legate::Type::Code::BOOL) {
      // Convert to bit packed
      data = null_mask_bools_to_bits(physical_array.data());
    } else {
      data = ARROW_RESULT(arrow::AllocateBuffer(nbytes * sizeof(int8_t)));
      std::memcpy(data->mutable_data(), read_accessor_as_1d_bytes(physical_array.data()), nbytes);
    }
    std::shared_ptr<arrow::Buffer> null_bitmask;
    if (array_->nullable()) { null_bitmask = null_mask_bools_to_bits(physical_array.null_mask()); }
    auto array_data = arrow::ArrayData::Make(arrow_type_, num_rows(), {null_bitmask, data});
    return arrow::MakeArray(array_data);
  }
}

LogicalColumn LogicalColumn::slice(const legate::Slice& slice) const
{
  return LogicalColumn(array_->slice(0, slice), arrow_type_);
}

std::string LogicalColumn::repr(size_t max_num_items) const
{
  std::stringstream ss;
  ss << "LogicalColumn(";
  if (unbound()) {
    ss << "data=unbound, ";
    if (array_->nullable()) { ss << "null_mask=unbound, "; }
    ss << "dtype=" << array_->type();
  } else {
    legate::PhysicalArray ary = array_->get_physical_array();

    // Notice, `get_physical_array()` returns host memory always
    ss << legate::dataframe::repr(
      ary, max_num_items, legate::Memory::Kind::SYSTEM_MEM, cudaStream_t{0});
  }
  if (unbound() || num_rows() == 1) { ss << ", is_scalar=" << (is_scalar() ? "True" : "False"); }
  ss << ")";
  return ss.str();
}

namespace task {

std::string PhysicalColumn::repr(legate::Memory::Kind mem_kind,
                                 cudaStream_t stream,
                                 size_t max_num_items) const
{
  std::stringstream ss;
  ss << "PhysicalColumn(";
  ss << legate::dataframe::repr(array_, max_num_items, mem_kind, stream) << ")";
  return ss.str();
}

std::shared_ptr<arrow::Array> PhysicalColumn::arrow_array_view() const
{
  if (unbound()) {
    throw std::runtime_error(
      "Cannot call `.arrow_array()` on a unbound LogicalColumn, please bind it using "
      "`.move_into()`");
  }
  if (array_.nested()) {
    if (array_.type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_.as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      auto num_chars                      = chars.data().shape<1>().volume();
      // Its possible to have an empty string, in which we want to avoid giving arrow a null ptr
      std::shared_ptr<arrow::Buffer> data;
      if (num_chars == 0) {
        data = ARROW_RESULT(arrow::AllocateBuffer(1));
      } else {
        data = std::make_shared<arrow::Buffer>(
          reinterpret_cast<const uint8_t*>(read_accessor_as_1d_bytes(chars)), num_chars);
      }

      std::shared_ptr<arrow::Buffer> null_bitmask;
      if (a.nullable()) { null_bitmask = null_mask_bools_to_bits(array_.null_mask()); }

      auto offsets = global_ranges_to_arrow_offsets(a.ranges().data());

      return std::make_shared<arrow::StringArray>(num_rows(), offsets, data, null_bitmask);

    } else {
      throw std::invalid_argument("nested dtype " + array_.type().to_string() + " isn't supported");
    }
  } else {
    auto nbytes = array_.shape<1>().volume() * array_.type().size();
    // 1. Create arrow data buffer - try to use the existing data
    std::shared_ptr<arrow::Buffer> buffer;
    if (this->type().code() == legate::Type::Code::BOOL) {
      // Arrow stores bool bit packed so we must copy
      buffer = null_mask_bools_to_bits(array_.data());
    } else {
      // For other types, we can use the existing data directly
      buffer = std::make_shared<arrow::Buffer>(
        reinterpret_cast<const uint8_t*>(read_accessor_as_1d_bytes(array_.data())), nbytes);
    }

    // 2. Handle null mask
    std::shared_ptr<arrow::Buffer> null_bitmask;
    if (array_.nullable()) { null_bitmask = null_mask_bools_to_bits(array_.null_mask()); }
    // 3. Create ArrayData from buffer
    auto array_data = arrow::ArrayData::Make(arrow_type_, num_rows(), {null_bitmask, buffer});
    return arrow::MakeArray(array_data);
  }
}

void PhysicalColumn::copy_into(std::shared_ptr<arrow::Array> column)
{
  // String columns seem tricky, so only check their data for being unbound.
  if (unbound()) {
    throw std::invalid_argument("Cannot call `.copy_into()` on an unbound column.");
  }
  // TODO: this copies the data, we ideally want to move the arrow buffer.
  detail::from_arrow(array_, column, scalar_out_);
}

void PhysicalColumn::move_into(std::shared_ptr<arrow::Array> column)
{
  if (!unbound()) { throw std::invalid_argument("Cannot call `.move_into()` on a bound column."); }
  // TODO: this copies the data, we ideally want to move the arrow buffer.
  detail::from_arrow(array_, column, scalar_out_);
}

void PhysicalColumn::bind_empty_data() const
{
  if (!unbound()) {
    throw std::invalid_argument("Cannot call `.bind_empty_data()` on a bound column");
  }

  if (scalar_out_) {
    throw std::logic_error("Binding empty data to scalar column should not happen?");
  }

  if (array_.nullable()) { array_.null_mask().bind_empty_data(); }
  if (array_.nested()) {
    legate::StringPhysicalArray ary = array_.as_string_array();
    ary.ranges().data().bind_empty_data();
    ary.chars().data().bind_empty_data();
  } else {
    array_.data().bind_empty_data();
  }
}

}  // namespace task

namespace argument {

legate::Variable add_next_input(legate::AutoTask& task, const LogicalColumn& col, bool broadcast)
{
  add_next_scalar_vector(task, serialize_arrow_type(col.arrow_type()));
  auto arr      = col.get_logical_array();
  auto variable = task.add_input(arr);
  if (broadcast) { task.add_constraint(legate::broadcast(variable, {0})); }
  return variable;
}

legate::Variable add_next_output(legate::AutoTask& task, const LogicalColumn& col)
{
  add_next_scalar_vector(task, serialize_arrow_type(col.arrow_type()));
  // While we don't care much for reading from a scalar column, pass scalar information
  // for outputs to enforce the result having the right size.
  add_next_scalar(task, col.is_scalar());
  auto variable = task.add_output(col.get_logical_array());
  // Output scalars must be broadcast (for inputs alignment should enforce reasonable things).
  // (If needed, we could enforce that only rank 0 can bind a result instead.)
  if (col.is_scalar()) { task.add_constraint(legate::broadcast(variable, {0})); }
  return variable;
}

}  // namespace argument

}  // namespace legate::dataframe
