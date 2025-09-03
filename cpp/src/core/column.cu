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

#include <cuda_runtime_api.h>

#include <legate/cuda/cuda.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
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
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(
      data_ptr, source_ptr, num_rows * sizeof(T), cudaMemcpyDeviceToDevice, stream));
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

  if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    /*
     * NOTE: We can probably remove this eventually, it exists currently
     * mainly because a lot of tests are still written using cudf types and
     * run for CPU only as well.
     * I.e. if we can't copy to GPU, copy to CPU via arrow.
     */
    auto device_array = cudf::to_arrow_host(col, stream);
    auto arrow_type   = to_arrow_type(col.type().id());
    auto arrow_array  = ARROW_RESULT(arrow::ImportArray(&device_array->array, arrow_type));
    return detail::from_arrow(arrow_array);
  }

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

}  // namespace

LogicalColumn::LogicalColumn(cudf::column_view cudf_col, rmm::cuda_stream_view stream)
  : LogicalColumn{from_cudf(cudf_col, stream), cudf_col.type(), /* scalar */ false}
{
}

LogicalColumn::LogicalColumn(const cudf::scalar& cudf_scalar, rmm::cuda_stream_view stream)
  : LogicalColumn{from_cudf(cudf_scalar, stream), cudf_scalar.type(), /* scalar */ true}
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
    this->cudf_type(),
    num_rows(),
    copy_physical_array_to_device(array_->get_physical_array(), stream),
    std::move(null_mask),
    null_count);
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

namespace task {

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
  return cudf::column_view(
    this->cudf_type(), num_rows(), data, null_mask, null_count, offset, children);
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

void PhysicalColumn::copy_into(const cudf::column_view& column)
{
  // String columns seem tricky, so only check their data for being unbound.
  if (unbound()) {
    throw std::invalid_argument("Cannot call `.copy_into()` on an unbound column.");
  }
  from_cudf(array_, column, ctx_->stream(), ctx_->mr(), scalar_out_);
}

void PhysicalColumn::copy_into(std::unique_ptr<cudf::scalar> scalar)
{
  // NOTE: this goes via a column-view.  Moving data more directly may be
  // preferable (although libcudf could also grow a way to get a column view).
  auto col = cudf::make_column_from_scalar(*scalar, 1, ctx_->stream());
  copy_into(std::move(col));
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

}  // namespace task
}  // namespace legate::dataframe
