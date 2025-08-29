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

#include <arrow/api.h>
#ifdef LEGATE_DATAFRAME_USE_CUDA
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#endif

#include <legate.h>

namespace legate::dataframe {

#ifdef LEGATE_DATAFRAME_USE_CUDA
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
#endif

/**
 * @brief Converts a boolean null mask stored in a PhysicalStore to an Arrow bit-packed buffer.
 *
 * This function takes a PhysicalStore containing boolean values representing a null mask,
 * and converts it into a bit-packed Arrow buffer, where each bit corresponds to the validity
 * of a value (1 for valid, 0 for null).
 *
 * @param bools The PhysicalStore containing boolean values representing the null mask.
 * @return A shared pointer to an Arrow buffer containing the bit-packed null mask.
 */
[[nodiscard]] std::shared_ptr<arrow::Buffer> null_mask_bools_to_bits(
  const legate::PhysicalStore& bools);

#ifdef LEGATE_DATAFRAME_USE_CUDA
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
#endif

}  // namespace legate::dataframe
