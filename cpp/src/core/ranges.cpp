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

#include <legate_dataframe/core/ranges.hpp>
#include <legate_dataframe/utils.hpp>
#include <limits>

namespace legate::dataframe {

std::shared_ptr<arrow::Buffer> global_ranges_to_arrow_offsets(const legate::PhysicalStore& ranges)
{
  using offset_type = typename arrow::StringArray::TypeClass::offset_type;
  std::shared_ptr<arrow::Buffer> offsets =
    ARROW_RESULT(arrow::AllocateBuffer((ranges.shape<1>().volume() + 1) * sizeof(offset_type)));
  auto offsets_ptr = reinterpret_cast<offset_type*>(offsets->mutable_data());
  auto ranges_ptr  = ranges.read_accessor<legate::Rect<1>, 1>().ptr(ranges.shape<1>().lo[0]);
  auto ranges_size = ranges.shape<1>().volume();
  if (ranges_size == 0) {
    offsets_ptr[0] = 0;
    return offsets;
  };
  auto global_range_offset = ranges_ptr[0].lo[0];
  for (size_t i = 0; i < ranges_size; ++i) {
    offsets_ptr[i] = ranges_ptr[i].lo[0] - global_range_offset;
  }
  offsets_ptr[ranges_size] = ranges_ptr[ranges_size - 1].hi[0] - global_range_offset + 1;
  return offsets;
}

void arrow_offsets_to_local_ranges(const arrow::StringArray& array, legate::Rect<1>* ranges_acc)
{
  auto first_offset = array.value_offset(0);
  if (first_offset == 0) {
    for (size_t i = 0; i < array.length(); ++i) {
      ranges_acc[i].lo[0] = array.value_offset(i);
      ranges_acc[i].hi[0] = array.value_offset(i + 1) - 1;
    }
  } else {
    for (size_t i = 0; i < array.length(); ++i) {
      ranges_acc[i].lo[0] = array.value_offset(i) - first_offset;
      ranges_acc[i].hi[0] = array.value_offset(i + 1) - 1 - first_offset;
    }
  }
}

void arrow_offsets_to_local_ranges(const arrow::LargeStringArray& array,
                                   legate::Rect<1>* ranges_acc)
{
  auto first_offset = array.value_offset(0);
  if (first_offset == 0) {
    for (size_t i = 0; i < array.length(); ++i) {
      ranges_acc[i].lo[0] = array.value_offset(i);
      ranges_acc[i].hi[0] = array.value_offset(i + 1) - 1;
    }
  } else {
    for (size_t i = 0; i < array.length(); ++i) {
      ranges_acc[i].lo[0] = array.value_offset(i) - first_offset;
      ranges_acc[i].hi[0] = array.value_offset(i + 1) - 1 - first_offset;
    }
  }
}

}  // namespace legate::dataframe
