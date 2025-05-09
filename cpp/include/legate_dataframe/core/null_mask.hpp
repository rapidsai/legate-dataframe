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

#pragma once

#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <legate.h>

namespace legate::dataframe {

/**
 * @brief Convert a null mask of booleans (legate) to bits (cudf)
 *
 * @param bools The boolean store.
 * @param mem_kind The memory type of `ranges`.
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource to use for all device memory allocations.
 * @return The cudf bitmask.
 */
[[nodiscard]] rmm::device_buffer null_mask_bools_to_bits(const legate::PhysicalStore& bools,
                                                         legate::Memory::Kind mem_kind,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr);

[[nodiscard]] std::shared_ptr<arrow::Buffer> null_mask_bools_to_bits(
  const legate::PhysicalStore& bools);
/**
 * @brief Convert a null mask of bits (cudf) to booleans (legate)
 *
 * @param bools_size The size of `bools`.
 * @param bools The output array of booleans.
 * @param bitmask The cudf bitmask.
 * @param stream CUDA stream used for device memory operations.
 */
void null_mask_bits_to_bools(int64_t bools_size,
                             bool* bools,
                             const cudf::bitmask_type* bitmask,
                             rmm::cuda_stream_view stream);

}  // namespace legate::dataframe
