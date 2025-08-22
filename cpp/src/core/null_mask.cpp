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

#include <arrow/api.h>

#include <legate_dataframe/core/null_mask.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

std::shared_ptr<arrow::Buffer> null_mask_bools_to_bits(const legate::PhysicalStore& bools)
{
  auto bools_acc   = bools.read_accessor<bool, 1>();
  auto bools_shape = bools.shape<1>();
  auto bools_size  = bools_shape.volume();
  auto bitmap_size = arrow::bit_util::BytesForBits(bools_size);
  auto buffer      = ARROW_RESULT(arrow::AllocateBuffer(bitmap_size));
  auto ptr         = buffer->mutable_data();
  std::memset(ptr, 0, static_cast<size_t>(buffer->capacity()));
  for (size_t i = 0; i < bools_size; ++i) {
    if (bools_acc[bools_shape.lo[0] + i] > 0) { arrow::bit_util::SetBit(ptr, i); }
  }
  return buffer;
}

}  // namespace legate::dataframe
