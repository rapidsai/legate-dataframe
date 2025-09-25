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

#include <limits>

#include <cuda_runtime_api.h>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <legate_dataframe/core/null_mask.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {

__device__ constexpr auto max_bitmask_type = std::numeric_limits<cudf::bitmask_type>::max();

/**
 * @brief CUDA kernel to convert booleans into bits
 *
 * Each CUDA thread writes a whole bitmask word by first setting all bits to 1
 * and then clear the bits that maps to false values in `bools_acc`
 */
template <typename RangesAcc>
__global__ void bools_to_bitmask(int64_t bools_size,
                                 legate::Point<1> bools_lo,
                                 cudf::bitmask_type* bitmask,
                                 RangesAcc bools_acc)
{
  constexpr auto wordsize = cudf::detail::size_in_bits<cudf::bitmask_type>();
  auto word_id            = blockIdx.x * blockDim.x + threadIdx.x;
  auto start_bit          = word_id * wordsize;
  if (start_bit < bools_size) {
    bitmask[word_id] = max_bitmask_type;
    for (auto i = start_bit; i < start_bit + wordsize && i < bools_size; ++i) {
      if (!bools_acc[bools_lo + i]) { cudf::clear_bit_unsafe(bitmask, i); }
    }
  }
}
}  // namespace

[[nodiscard]] rmm::device_buffer null_mask_bools_to_bits(const legate::PhysicalStore& bools,
                                                         legate::Memory::Kind mem_kind,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  auto bools_acc   = bools.read_accessor<bool, 1>();
  auto bools_shape = bools.shape<1>();
  auto bools_size  = bools_shape.volume();

  rmm::device_buffer bitmask =
    cudf::create_null_mask(bools_size, cudf::mask_state::UNINITIALIZED, stream, mr);

  if (bools_size == 0) { return bitmask; }

  // Launch one CUDA thread per bitmask word.
  const int threads_per_block = 256;
  auto num_blocks =
    (cudf::num_bitmask_words(bools_size) + threads_per_block - 1) / threads_per_block;
  if (is_device_mem(mem_kind)) {
    bools_to_bitmask<<<num_blocks, threads_per_block, 0, stream>>>(
      bools_size, bools_shape.lo, static_cast<cudf::bitmask_type*>(bitmask.data()), bools_acc);
  } else {
    auto tmp_dev_buf      = rmm::device_buffer(bools_size * sizeof(bool), stream, mr);
    auto bools_acc_on_dev = static_cast<bool*>(tmp_dev_buf.data());
    LDF_CUDA_TRY(cudaMemcpyAsync(bools_acc_on_dev,
                                 bools_acc.ptr(0),
                                 bools_size * sizeof(bool),
                                 cudaMemcpyHostToDevice,
                                 stream));
    bools_to_bitmask<<<num_blocks, threads_per_block, 0, stream>>>(
      bools_size, 0, static_cast<cudf::bitmask_type*>(bitmask.data()), bools_acc_on_dev);

    LDF_CUDA_TRY(cudaStreamSynchronize(stream));
  }
  return bitmask;
}

namespace {
/**
 * @brief CUDA kernel to convert bits into booleans
 *
 * Each CUDA thread writes a boolean by reading the corresponding bit
 */
__global__ void bitmask_to_bools(int64_t bools_size, bool* bools, const cudf::bitmask_type* bitmask)
{
  auto tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= bools_size) return;
  bools[tid] = cudf::bit_is_set(bitmask, tid);
}
}  // namespace

void null_mask_bits_to_bools(int64_t bools_size,
                             bool* bools,
                             const cudf::bitmask_type* bitmask,
                             rmm::cuda_stream_view stream)
{
  const int threads_per_block = 256;
  auto num_blocks             = (bools_size + threads_per_block - 1) / threads_per_block;
  bitmask_to_bools<<<num_blocks, threads_per_block, 0, stream>>>(bools_size, bools, bitmask);
}

}  // namespace legate::dataframe
