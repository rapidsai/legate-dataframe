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

#include <cuda_runtime_api.h>

#include <legate/cuda/cuda.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_uvector.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/null_mask.hpp>
#include <legate_dataframe/core/print.hpp>
#include <legate_dataframe/core/ranges.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

template <typename T>
T* maybe_bind_buffer(legate::PhysicalStore store, std::size_t size)
{
  T* out;
  if (store.is_unbound_store()) {
    out = store.create_output_buffer<T, 1>(legate::Point<1>(size), true).ptr(0);
  } else {
    auto acc = store.write_accessor<T, 1>();
    assert((store.shape<1>().hi[0] - store.shape<1>().lo[0]) == -1 ||
           acc.accessor.is_dense_row_major(store.shape<1>()));
    assert((store.shape<1>().hi[0] + 1 - store.shape<1>().lo[0]) == size);
    out = acc.ptr(store.shape<1>().lo[0]);
  }
  return out;
}

struct move_into_fn {
  template <typename T, std::enable_if_t<cudf::is_rep_layout_compatible<T>()>* = nullptr>
  void operator()(legate::PhysicalArray& array,
                  const cudf::column_view& column,
                  cudaStream_t stream,
                  TaskMemoryResource* mr)
  {
    const auto num_rows = column.size();
    if (array.nullable()) {
      bool* null_mask_ptr = maybe_bind_buffer<bool>(array.null_mask(), num_rows);

      if (column.nullable()) {
        null_mask_bits_to_bools(num_rows, null_mask_ptr, column.null_mask(), stream);
      } else {
        LEGATE_CHECK_CUDA(cudaMemsetAsync(
          null_mask_ptr, std::numeric_limits<bool>::max(), num_rows * sizeof(bool), stream));
      }
    }

    if (mr != nullptr) {
      auto mem_alloc = mr->release_buffer(column);
      if (mem_alloc.valid() && array.data().is_unbound_store()) {
        array.data().bind_untyped_data(mem_alloc.buffer(), num_rows);
        return;
      }
    }

    T* data_ptr         = maybe_bind_buffer<T>(array.data(), num_rows);
    const T* source_ptr = column.data<T>();
    // cudaPointerAttributes attr;
    // LEGATE_CHECK_CUDA(cudaPointerGetAttributes(&attr, data_ptr));
    // LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream));
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(
      data_ptr, source_ptr, num_rows * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    // LEGATE_CHECK_CUDA(cudaDeviceSynchronize());
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, cudf::string_view>>* = nullptr>
  void operator()(legate::PhysicalArray& array,
                  const cudf::column_view& column,
                  cudaStream_t stream,
                  TaskMemoryResource* mr)
  {
    // The string version currently doesn't support already bound chars outputs.  Presumably, this
    // can't happen right now anyway, because the result size is not fixed so it should fail early?
    const auto num_rows = column.size();

    if (array.nullable()) {
      bool* null_mask_ptr = maybe_bind_buffer<bool>(array.null_mask(), num_rows);

      if (column.nullable()) {
        null_mask_bits_to_bools(num_rows, null_mask_ptr, column.null_mask(), stream);
      } else {
        LEGATE_CHECK_CUDA(cudaMemsetAsync(
          null_mask_ptr, std::numeric_limits<bool>::max(), num_rows * sizeof(bool), stream));
      }
    }

    cudf::strings_column_view str_col{column};
    legate::StringPhysicalArray ary = array.as_string_array();

    if (str_col.size() == 0) {
      if (ary.ranges().data().is_unbound_store()) { ary.ranges().data().bind_empty_data(); }
      if (ary.chars().data().is_unbound_store()) { ary.chars().data().bind_empty_data(); }
      return;
    }

    auto ranges_size = str_col.offsets().size() - 1;
    auto ranges_ptr  = maybe_bind_buffer<legate::Rect<1>>(ary.ranges().data(), ranges_size);
    cudf_offsets_to_local_ranges(ranges_size, ranges_ptr, str_col.offsets(), stream);

    if (str_col.offsets().offset() != 0) {
      throw std::runtime_error("string column seems sliced, which is currently not supported.");
    }
    auto nbytes = str_col.chars_size(stream);
    // NOTE: a string array can never have it's chars data already bound (size may change).
    if (mr != nullptr) {
      // If valid allocation, don't copy the string data.
      auto mem_alloc = mr->release_buffer(str_col, stream);
      if (mem_alloc.valid() && ary.chars().data().is_unbound_store()) {
        ary.chars().data().bind_untyped_data(mem_alloc.buffer(), nbytes);
        return;
      }
    }

    auto chars_ptr = maybe_bind_buffer<int8_t>(ary.chars().data(), nbytes);
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(
      chars_ptr, str_col.chars_begin(stream), nbytes, cudaMemcpyDeviceToDevice, stream));
  }

  template <typename T,
            std::enable_if_t<!(cudf::is_rep_layout_compatible<T>() ||
                               std::is_same_v<T, cudf::string_view>)>* = nullptr>
  void operator()(legate::PhysicalArray& array,
                  const cudf::column_view& column,
                  cudaStream_t stream,
                  TaskMemoryResource* mr)
  {
    // TODO: support lists
    throw std::invalid_argument("move_into(): type not supported");
  }
};

/*
 * Helper to either bind or copy cudf data into PhysicalArray.
 * Context may be `nullptr` when run outside of a trask.
 * This function may take possession of the column data (and bind it to the array).
 */
void from_cudf(legate::PhysicalArray array,
               const cudf::column_view& column,
               cudaStream_t stream,
               TaskMemoryResource* mr = nullptr,
               bool scalar            = false)
{
  // Expect the types to match
  if (array.type() != to_legate_type(column.type().id())) {
    throw std::invalid_argument("from_cudf(): type mismatch.");
  }
  // NOTE(seberg): In some cases (replace nulls) we expect no nulls, but
  //     seem to get a nullable column.  So also check `has_nulls()`.
  if (column.nullable() && !array.nullable() && column.has_nulls()) {
    throw std::invalid_argument(
      "from_cudf(): the cudf column is nullable while the PhysicalArray isn't");
  }

  if (scalar && column.size() != 1) {
    throw std::invalid_argument("from_cudf(): scalar column must have size one.");
  }
  cudf::type_dispatcher(column.type(), move_into_fn{}, array, column, stream, mr);
}

legate::LogicalArray from_cudf(const cudf::column_view& col, rmm::cuda_stream_view stream)
{
  auto runtime = legate::Runtime::get_runtime();

  auto cudf_nullable = col.nullable();  // could also count nulls
  if (cudf::type_id::STRING == col.type().id()) {
    cudf::strings_column_view str_col{col};
    auto nbytes = str_col.chars_size(stream);
    auto array  = runtime->create_string_array(
      runtime->create_array({std::uint64_t(col.size())}, legate::rect_type(1), cudf_nullable),
      runtime->create_array({std::uint64_t(nbytes)}, legate::int8()));
    from_cudf(array.get_physical_array(legate::mapping::StoreTarget::FBMEM), col, stream);
    return array;
  }
  if (col.num_children() > 0) {
    throw std::invalid_argument("non-string column with children isn't supported");
  }
  auto array = runtime->create_array({std::uint64_t(col.size())},
                                     to_legate_type(col.type().id()),
                                     cudf_nullable,
                                     false /* scalar */);
  from_cudf(array.get_physical_array(legate::mapping::StoreTarget::FBMEM), col, stream);
  return array;
}

legate::LogicalArray from_cudf(const cudf::scalar& scalar, rmm::cuda_stream_view stream)
{
  // NOTE: this goes via a column-view.  Moving data more directly may be
  // preferable (although libcudf could also grow a way to get a column view).
  auto col = cudf::make_column_from_scalar(scalar, 1, stream);
  return from_cudf(col->view(), stream);
}

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
legate::LogicalArray from_arrow(std::shared_ptr<arrow::Array> arrow_array, bool scalar = false)
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
}  // namespace

LogicalColumn::LogicalColumn(cudf::column_view cudf_col, rmm::cuda_stream_view stream)
  : LogicalColumn{from_cudf(cudf_col, stream), cudf_col.type(), /* scalar */ false}
{
}

LogicalColumn::LogicalColumn(const cudf::scalar& cudf_scalar, rmm::cuda_stream_view stream)
  : LogicalColumn{from_cudf(cudf_scalar, stream), cudf_scalar.type(), /* scalar */ true}
{
}

LogicalColumn::LogicalColumn(std::shared_ptr<arrow::Array> arrow_array)
  : LogicalColumn{// This type conversion monstrosity can be improved
                  from_arrow(arrow_array),
                  to_cudf_type(arrow_array->type()),
                  /* scalar */ false}
{
}

LogicalColumn::LogicalColumn(std::shared_ptr<arrow::Scalar> arrow_scalar)
  : LogicalColumn{// This type conversion monstrosity can be improved
                  from_arrow(arrow_scalar),
                  to_cudf_type(arrow_scalar->type),
                  /* scalar */ true}
{
}

namespace {

/**
 * @brief Since Legate's get_physical_array() doesn't support device memory, use this function to
 * copy a physical array to device.
 * TODO: If `get_physical_array()` supports device memory this can be replaced.
 */
[[nodiscard]] rmm::device_buffer copy_physical_array_to_device(const PhysicalArray& physical_array,
                                                               cudaStream_t stream)
{
  auto host_ary_nbytes = physical_array.shape<1>().volume() * physical_array.type().size();
  auto ret             = rmm::device_buffer(host_ary_nbytes, stream);
  LEGATE_CHECK_CUDA(cudaMemcpyAsync(ret.data(),
                                    read_accessor_as_1d_bytes(physical_array.data()),
                                    host_ary_nbytes,
                                    cudaMemcpyHostToDevice,
                                    stream));
  return ret;
}

}  // namespace

std::unique_ptr<cudf::column> LogicalColumn::get_cudf(rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr) const
{
  // TODO(seberg): This function goes via system memory but could use FBMEM in get_physical_array()
  // One way to achieve this may be to refactor PhysicalColumn::column_view() into a helper
  // and reuse that here. (Once we have a column view, copying it seems reasonable.)
  if (array_->nested()) {
    if (array_->type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_->get_physical_array().as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      const auto num_chars                = chars.data().shape<1>().volume();

      // Copy and convert the physical array of ranges to a new cudf column
      std::unique_ptr<cudf::column> cudf_offsets = global_ranges_to_cudf_offsets(
        a.ranges(), num_chars, legate::Memory::Kind::SYSTEM_MEM, stream, mr);

      // Copy the physical array of chars to a new cudf column
      auto chars_buf = copy_physical_array_to_device(chars, stream);
      rmm::device_buffer null_mask{};
      cudf::size_type null_count{0};
      if (a.nullable()) {
        null_mask =
          null_mask_bools_to_bits(a.null_mask(), legate::Memory::Kind::SYSTEM_MEM, stream, mr);
        null_count =
          cudf::null_count(static_cast<const cudf::bitmask_type*>(null_mask.data()), 0, num_rows());
      }
      // Create a new string column from ranges and chars
      return cudf::make_strings_column(num_rows(),
                                       std::move(cudf_offsets),
                                       std::move(chars_buf),
                                       null_count,
                                       std::move(null_mask));
    } else {
      throw std::invalid_argument("nested dtype " + array_->type().to_string() +
                                  " isn't supported");
    }
  }
  rmm::device_buffer null_mask{};
  cudf::size_type null_count{0};
  if (array_->nullable()) {
    legate::PhysicalArray ary = array_->get_physical_array();
    null_mask =
      null_mask_bools_to_bits(ary.null_mask(), legate::Memory::Kind::SYSTEM_MEM, stream, mr);
    null_count =
      cudf::null_count(static_cast<const cudf::bitmask_type*>(null_mask.data()), 0, num_rows());
  }
  return std::make_unique<cudf::column>(
    cudf_type_,
    num_rows(),
    copy_physical_array_to_device(array_->get_physical_array(), stream),
    std::move(null_mask),
    null_count);
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
    auto array_data =
      arrow::ArrayData::Make(to_arrow_type(cudf_type_.id()), num_rows(), {null_bitmask, data});
    return arrow::MakeArray(array_data);
  }
}

std::unique_ptr<cudf::scalar> LogicalColumn::get_cudf_scalar(
  rmm::cuda_stream_view stream, rmm::mr::device_memory_resource* mr) const
{
  // NOTE: We could specialize simple scalars here at least.
  auto col = get_cudf(stream, mr);
  if (col->size() != 1) {
    throw std::invalid_argument("only length 1/scalar columns can be converted to scalar.");
  }
  return std::move(cudf::get_element(col->view(), 0));
}

LogicalColumn LogicalColumn::slice(const legate::Slice& slice) const
{
  return LogicalColumn(array_->slice(0, slice), cudf_type_);
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

cudf::column_view PhysicalColumn::column_view() const
{
  if (unbound()) {
    throw std::runtime_error(
      "Cannot call `.column_view()` on a unbound LogicalColumn, please bind it using "
      "`.move_into()`");
  }

  const void* data                    = nullptr;
  const cudf::bitmask_type* null_mask = nullptr;
  cudf::size_type null_count          = 0;
  cudf::size_type offset              = 0;
  std::vector<cudf::column_view> children;

  if (array_.nested()) {
    if (array_.type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = array_.as_string_array();
      const legate::PhysicalArray chars   = a.chars();
      const auto num_chars                = chars.data().shape<1>().volume();

      std::unique_ptr<cudf::column> cudf_offsets = global_ranges_to_cudf_offsets(
        a.ranges(), num_chars, legate::Memory::Kind::GPU_FB_MEM, ctx_->stream(), ctx_->mr());

      // To keep the offsets alive beyond this function, we push it to temporaries before
      // adding it as the first child.
      tmp_cols_.push_back(std::move(cudf_offsets));
      children.push_back(tmp_cols_.back()->view());

      // The second child is the character column
      data = read_accessor_as_1d_bytes(chars.data());
    } else {
      throw std::invalid_argument("nested dtype " + array_.type().to_string() + " isn't supported");
    }
  } else {
    data = read_accessor_as_1d_bytes(array_.data());
  }
  if (array_.nullable()) {
    tmp_null_masks_.push_back(null_mask_bools_to_bits(
      array_.null_mask(), legate::Memory::Kind::GPU_FB_MEM, ctx_->stream(), ctx_->mr()));
    null_mask  = static_cast<const cudf::bitmask_type*>(tmp_null_masks_.back().data());
    null_count = cudf::null_count(null_mask, 0, num_rows(), ctx_->stream());
  }
  return cudf::column_view(cudf_type_, num_rows(), data, null_mask, null_count, offset, children);
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
    auto array_data =
      arrow::ArrayData::Make(to_arrow_type(cudf_type_.id()), num_rows(), {null_bitmask, buffer});
    return arrow::MakeArray(array_data);
  }
}

std::unique_ptr<cudf::scalar> PhysicalColumn::cudf_scalar() const
{
  if (num_rows() != 1) {
    throw std::invalid_argument("can only convert length one columns to scalar.");
  }
  return cudf::get_element(column_view(), 0);
}

void PhysicalColumn::copy_into(std::unique_ptr<cudf::column> column)
{
  // String columns seem tricky, so only check their data for being unbound.
  if (unbound()) {
    throw std::invalid_argument("Cannot call `.copy_into()` on an unbound column.");
  }
  from_cudf(array_, column->view(), ctx_->stream(), ctx_->mr(), scalar_out_);
}

void PhysicalColumn::copy_into(std::unique_ptr<cudf::scalar> scalar)
{
  // NOTE: this goes via a column-view.  Moving data more directly may be
  // preferable (although libcudf could also grow a way to get a column view).
  auto col = cudf::make_column_from_scalar(*scalar, 1, ctx_->stream());
  copy_into(std::move(col));
}

void PhysicalColumn::copy_into(std::shared_ptr<arrow::Array> column)
{
  // String columns seem tricky, so only check their data for being unbound.
  if (unbound()) {
    throw std::invalid_argument("Cannot call `.copy_into()` on an unbound column.");
  }
  // TODO: this copies the data, we ideally want to move the arrow buffer.
  from_arrow(array_, column, scalar_out_);
}

void PhysicalColumn::move_into(std::unique_ptr<cudf::column> column)
{
  if (!unbound()) { throw std::invalid_argument("Cannot call `.move_into()` on a bound column."); }
  from_cudf(array_, column->view(), ctx_->stream(), ctx_->mr(), scalar_out_);
}

void PhysicalColumn::move_into(std::unique_ptr<cudf::scalar> scalar)
{
  // NOTE: this goes via a column-view.  Moving data more directly may be
  // preferable (although libcudf could also grow a way to get a column view).

  auto col = cudf::make_column_from_scalar(*scalar, 1, ctx_->stream());
  move_into(std::move(col));
}

void PhysicalColumn::move_into(std::shared_ptr<arrow::Array> column)
{
  if (!unbound()) { throw std::invalid_argument("Cannot call `.move_into()` on a bound column."); }
  // TODO: this copies the data, we ideally want to move the arrow buffer.
  from_arrow(array_, column, scalar_out_);
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
  add_next_scalar(task, static_cast<std::underlying_type_t<cudf::type_id>>(col.cudf_type().id()));
  auto arr      = col.get_logical_array();
  auto variable = task.add_input(arr);
  if (broadcast) { task.add_constraint(legate::broadcast(variable, {0})); }
  return variable;
}

legate::Variable add_next_output(legate::AutoTask& task, const LogicalColumn& col)
{
  add_next_scalar(task, static_cast<std::underlying_type_t<cudf::type_id>>(col.cudf_type().id()));
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
