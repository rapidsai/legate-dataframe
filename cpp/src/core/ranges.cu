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

#include <cuda_runtime_api.h>
#include <limits>

#include <legate/cuda/cuda.h>

#include <cudf/column/column_factories.hpp>

#include <legate_dataframe/core/ranges.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

/**
 * @brief CUDA kernel to convert ranges (legate) to offsets (cudf)
 */
template <typename RangesAcc, typename OffsetsAcc>
__global__ void ranges_to_offsets(int64_t offsets_size,
                                  int64_t vardata_size,
                                  legate::Point<1> ranges_shape_lo,
                                  RangesAcc ranges_acc,
                                  OffsetsAcc offsets_acc)
{
  auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid == offsets_size - 1) {
    offsets_acc[tid] = vardata_size;
  } else if (tid < offsets_size) {
    auto global_range_offset = ranges_acc[ranges_shape_lo].lo[0];
    offsets_acc[tid]         = ranges_acc[tid + ranges_shape_lo].lo[0] - global_range_offset;
  }
}

template <typename OffsetsAcc>
std::unique_ptr<cudf::column> global_ranges_to_cudf_offsets_impl(
  const legate::PhysicalArray ranges,
  int64_t num_chars,
  legate::Memory::Kind mem_kind,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  using RangeDType  = legate::Rect<1>;
  auto ranges_shape = ranges.data().shape<1>();
  auto ranges_size  = ranges_shape.volume();

  std::unique_ptr<cudf::column> cudf_offsets =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<OffsetsAcc>()},
                              ranges_size + 1,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  OffsetsAcc* offsets_acc = cudf_offsets->mutable_view().data<OffsetsAcc>();
  auto num_blocks =
    (cudf_offsets->size() + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;
  auto ranges_acc = ranges.data().read_accessor<RangeDType, 1>();

  if (is_device_mem(mem_kind)) {
    ranges_to_offsets<<<num_blocks, LEGATE_THREADS_PER_BLOCK, 0, stream>>>(
      cudf_offsets->size(), num_chars, ranges_shape.lo, ranges_acc, offsets_acc);
  } else {
    auto tmp_dev_buf       = rmm::device_buffer(ranges_size * sizeof(RangeDType), stream);
    auto ranges_acc_on_dev = static_cast<RangeDType*>(tmp_dev_buf.data());
    LEGATE_CHECK_CUDA(cudaMemcpyAsync(ranges_acc_on_dev,
                                      ranges_acc.ptr(0),
                                      ranges_size * sizeof(RangeDType),
                                      cudaMemcpyHostToDevice,
                                      stream));
    ranges_to_offsets<<<num_blocks, LEGATE_THREADS_PER_BLOCK, 0, stream>>>(
      cudf_offsets->size(), num_chars, 0, ranges_acc_on_dev, offsets_acc);
    LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  return cudf_offsets;
}

}  // namespace

std::unique_ptr<cudf::column> global_ranges_to_cudf_offsets(const legate::PhysicalArray ranges,
                                                            int64_t num_chars,
                                                            legate::Memory::Kind mem_kind,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::mr::device_memory_resource* mr)
{
  if (std::numeric_limits<int32_t>::max() >= num_chars) {
    return global_ranges_to_cudf_offsets_impl<int32_t>(ranges, num_chars, mem_kind, stream, mr);
  } else {
    return global_ranges_to_cudf_offsets_impl<int64_t>(ranges, num_chars, mem_kind, stream, mr);
  }
}

std::shared_ptr<arrow::Buffer> global_ranges_to_arrow_offsets(const legate::PhysicalStore& ranges)
{
  using offset_type = typename arrow::StringArray::TypeClass::offset_type;
  std::shared_ptr<arrow::Buffer> offsets =
    *arrow::AllocateBuffer((ranges.shape<1>().volume() + 1) * sizeof(offset_type));
  auto offsets_ptr         = reinterpret_cast<offset_type*>(offsets->mutable_data());
  auto ranges_ptr          = ranges.read_accessor<legate::Rect<1>, 1>().ptr(0);
  auto ranges_size         = ranges.shape<1>().volume();
  auto global_range_offset = ranges_ptr[0].lo[0];
  for (size_t i = 0; i < ranges_size; ++i) {
    offsets_ptr[i] = ranges_ptr[i].lo[0] - global_range_offset;
  }
  offsets_ptr[ranges_size] = ranges_ptr[ranges_size - 1].hi[0] - global_range_offset + 1;
  return offsets;
}

void arrow_offsets_to_local_ranges(const arrow::StringArray& array, legate::Rect<1>* ranges_acc)
{
  for (size_t i = 0; i < array.length(); ++i) {
    ranges_acc[i].lo[0] = array.value_offset(i);
    ranges_acc[i].hi[0] = array.value_offset(i + 1) - 1;
  }
}

namespace {
/**
 * @brief CUDA kernel to convert offsets (cudf) to ranges (legate)
 */
template <typename OffsetsAcc>
__global__ void offsets_to_ranges(int64_t ranges_size,
                                  legate::Rect<1>* ranges_acc,
                                  const OffsetsAcc* offsets_acc)
{
  auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= ranges_size) return;
  auto& range = ranges_acc[tid];
  range.lo[0] = offsets_acc[tid];
  range.hi[0] = offsets_acc[tid + 1] - 1;
}

}  // namespace

void cudf_offsets_to_local_ranges(int64_t ranges_size,
                                  legate::Rect<1>* ranges_acc,
                                  cudf::column_view offsets,
                                  rmm::cuda_stream_view stream)
{
  auto num_blocks = (ranges_size + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;

  if (offsets.type().id() == cudf::type_id::INT32) {
    offsets_to_ranges<<<num_blocks, LEGATE_THREADS_PER_BLOCK, 0, stream>>>(
      ranges_size, ranges_acc, offsets.data<int32_t>());
  } else {
    assert(offsets.type().id() == cudf::type_id::INT64);
    offsets_to_ranges<<<num_blocks, LEGATE_THREADS_PER_BLOCK, 0, stream>>>(
      ranges_size, ranges_acc, offsets.data<int64_t>());
  }
}

}  // namespace legate::dataframe
