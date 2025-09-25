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
#include <cudf/column/column.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#endif

#include <legate.h>

namespace legate::dataframe {

#ifdef LEGATE_DATAFRAME_USE_CUDA
/**
 * @brief Convert global ranges (legate) to local offsets (cudf)
 *
 * @param ranges The ranges relative to the global legate array.
 * @param num_chars The size of the corresponding character array.
 * @param mem_kind The memory type of `ranges`.
 * @param stream CUDA stream used for device memory operations.
 * @param mr Device memory resource to use for all device memory allocations.
 * @return The local offsets as a cudf column.
 */
[[nodiscard]] std::unique_ptr<cudf::column> global_ranges_to_cudf_offsets(
  const legate::PhysicalArray ranges,
  int64_t num_chars,
  legate::Memory::Kind mem_kind,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr);
#endif

/**
 * @brief Template function to convert global ranges to Arrow offsets.
 *
 * The caller should choose the appropriate offset for of the arrow string type.
 * @param ranges The global ranges to convert.
 * @return The Arrow offsets buffer.
 */
template <typename OffsetType>
[[nodiscard]] std::shared_ptr<arrow::Buffer> global_ranges_to_arrow_offsets(
  const legate::PhysicalStore& ranges);

/**
 * @brief Converts the offsets from an Arrow StringArray into local ranges.
 *
 * This function accounts for slicing and always returns an offset of 0 for the first string.
 *
 * @param array The Arrow StringArray containing the string data and offsets.
 * @param ranges_acc Pointer to an array of legate::Rect<1> where the computed ranges will be
 * stored.
 */
void arrow_offsets_to_local_ranges(const arrow::StringArray& array, legate::Rect<1>* ranges_acc);

/**
 * @brief Converts the offsets from an Arrow LargeStringArray into local ranges.
 *
 * This function accounts for slicing and always returns an offset of 0 for the first string.
 *
 * @param array The Arrow LargeStringArray containing the string data and offsets.
 * @param ranges_acc Pointer to an array of legate::Rect<1> where the computed ranges will be
 * stored.
 */
void arrow_offsets_to_local_ranges(const arrow::LargeStringArray& array,
                                   legate::Rect<1>* ranges_acc);

#ifdef LEGATE_DATAFRAME_USE_CUDA
/**
 * @brief Convert local offsets (cudf) to local ranges (legate)
 *
 * @param ranges_size The size of the local ranges accessed through `ranges_acc`.
 * @param ranges_acc The ranges write accessor.
 * @param offsets The offsets column (can be int32 or int64)
 * @param stream CUDA stream used for device memory operations.
 */
void cudf_offsets_to_local_ranges(int64_t ranges_size,
                                  legate::Rect<1>* ranges_acc,
                                  cudf::column_view offsets,
                                  rmm::cuda_stream_view stream);
#endif

}  // namespace legate::dataframe
