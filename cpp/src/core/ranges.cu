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
  OffsetsAcc* offsets_acc     = cudf_offsets->mutable_view().data<OffsetsAcc>();
  const int threads_per_block = 256;
  auto num_blocks             = (cudf_offsets->size() + threads_per_block - 1) / threads_per_block;
  auto ranges_acc             = ranges.data().read_accessor<RangeDType, 1>();

  if (is_device_mem(mem_kind)) {
    ranges_to_offsets<<<num_blocks, threads_per_block, 0, stream>>>(
      cudf_offsets->size(), num_chars, ranges_shape.lo, ranges_acc, offsets_acc);
  } else {
    auto tmp_dev_buf       = rmm::device_buffer(ranges_size * sizeof(RangeDType), stream, mr);
    auto ranges_acc_on_dev = static_cast<RangeDType*>(tmp_dev_buf.data());
    LDF_CUDA_TRY(cudaMemcpyAsync(ranges_acc_on_dev,
                                 ranges_acc.ptr(0),
                                 ranges_size * sizeof(RangeDType),
                                 cudaMemcpyHostToDevice,
                                 stream));
    ranges_to_offsets<<<num_blocks, threads_per_block, 0, stream>>>(
      cudf_offsets->size(), num_chars, 0, ranges_acc_on_dev, offsets_acc);
    LDF_CUDA_TRY(cudaStreamSynchronize(stream));
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
  const int threads_per_block = 256;
  auto num_blocks             = (ranges_size + threads_per_block - 1) / threads_per_block;

  if (offsets.type().id() == cudf::type_id::INT32) {
    offsets_to_ranges<<<num_blocks, threads_per_block, 0, stream>>>(
      ranges_size, ranges_acc, offsets.data<int32_t>());
  } else {
    assert(offsets.type().id() == cudf::type_id::INT64);
    offsets_to_ranges<<<num_blocks, threads_per_block, 0, stream>>>(
      ranges_size, ranges_acc, offsets.data<int64_t>());
  }
}

}  // namespace legate::dataframe
