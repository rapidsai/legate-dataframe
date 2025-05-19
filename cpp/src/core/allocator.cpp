/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <sstream>
#include <utility>

#include <legate/cuda/cuda.h>

#include <rmm/mr/device/per_device_resource.hpp>

#include <legate_dataframe/core/allocator.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

void* TaskMemoryResource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
{
  // std::cout << "TaskMR::do_allocate(" << this << ") - inside task: " << std::boolalpha
  //           << legate::is_running_in_task() << ", stream: " << stream.value()
  //           << " bytes: " << bytes << std::endl;
  if (bytes == 0) { return nullptr; }

  auto buffer       = legate::create_buffer<int8_t>(bytes, MEMORY_KIND, ALIGNMENT);
  auto ptr          = buffer.ptr(0);
  auto [_, success] = buffers_.emplace(std::make_pair(ptr, MemAlloc{bytes, std::move(buffer)}));
  if (!success) {
    throw std::runtime_error{"Invalid address for allocation: the address exist already!"};
  }
  return ptr;
}

void TaskMemoryResource::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
{
  // std::cout << "TaskMR::do_delocate(" << this << ") - inside task: " << std::boolalpha
  //           << legate::is_running_in_task() << ", stream: " << stream.value()
  //           << " bytes: " << bytes << std::endl;
  auto finder = buffers_.find(ptr);
  if (finder == buffers_.end()) { throw std::runtime_error{"Invalid address for deallocation"}; }
  if (finder->second.valid()) { finder->second.buffer().destroy(); }
  buffers_.erase(finder);
}

MemAlloc* TaskMemoryResource::find_buffer(const void* base_ptr)
{
  auto finder = buffers_.find(base_ptr);
  return finder == buffers_.end() ? nullptr : &finder->second;
}

MemAlloc TaskMemoryResource::release_buffer(cudf::column_view col)
{
  const void* col_ptr = col.data<int8_t>();
  MemAlloc* buffer    = find_buffer(col_ptr);
  if (col_ptr == nullptr || col.offset() != 0 || buffer == nullptr) {
    return MemAlloc{};
  } else {
    return std::move(*buffer);
  }
}

}  // namespace legate::dataframe
