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

MemAlloc TaskMemoryResource::release_buffer(cudf::strings_column_view col,
                                            rmm::cuda_stream_view stream)
{
  const void* col_ptr = col.chars_begin(stream);
  MemAlloc* buffer    = find_buffer(col_ptr);
  if (col_ptr == nullptr || buffer == nullptr) {
    return MemAlloc{};
  } else {
    return std::move(*buffer);
  }
}

void* GlobalMemoryResource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
{
  // std::cout << "GlobalMR::do_allocate(" << this << ") - inside task: " << std::boolalpha
  //           << legate::is_running_in_task() << ", stream: " << stream.value()
  //           << " bytes: " << bytes << std::endl;

  if (bytes == 0) { return nullptr; }

  if (legate::is_running_in_task()) {
    // Inside tasks, we can use the task RMM resource that is backed by legate buffers.
    return task_mr_.allocate(bytes, stream);
  } else {
    // Outside tasks (the legate control code task), legate device buffers is not available,
    // thus we have to use the CUDA memory resource that are backed by regular CUDA buffers.
    return cuda_mr_.allocate(bytes, stream);
  }
}

void GlobalMemoryResource::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
{
  // std::cout << "GlobalMR::do_deallocate(" << this << ") - inside task: " << std::boolalpha
  //           << legate::is_running_in_task() << ", stream: " << stream.value()
  //           << " bytes: " << bytes << std::endl;

  if (task_mr_.find_buffer(ptr) != nullptr) {
    return task_mr_.deallocate(ptr, bytes, stream);
  } else {
    return cuda_mr_.deallocate(ptr, bytes, stream);
  }
}

namespace {
std::vector<std::unique_ptr<GlobalMemoryResource>> _create_and_set_global_mrs()
{
  std::vector<std::unique_ptr<GlobalMemoryResource>> ret;
  const int num_gpus = rmm::get_num_cuda_devices();

  int current_device{-1};
  LEGATE_CHECK_CUDA(cudaGetDevice(&current_device));

  for (int i = 0; i < num_gpus; ++i) {
    LEGATE_CHECK_CUDA(cudaSetDevice(i));
    if (rmm::mr::detail::initial_resource() != rmm::mr::get_current_device_resource()) {
      throw std::runtime_error(
        "Legate-dataframe failed to set the current RMM memory resource because it has already "
        "been set, most likely by a call to `rmm::mr::set_per_device_resource()` or "
        "`rmm::mr::set_current_device_resource()`. To disable Legate-dataframe's use of the "
        "current RMM memory resource (not recommended), define the environment variable "
        "`LDF_DISABLE_GLOBAL_MEMORY_RESOURCE` on all nodes");
    }
    ret.push_back(std::make_unique<GlobalMemoryResource>());
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{i}, ret.back().get());
  }
  LEGATE_CHECK_CUDA(cudaSetDevice(current_device));
  return ret;
}
}  // namespace

void GlobalMemoryResource::set_as_default_mmr_resource()
{
  // By using a static variable, we make sure we only create the resources once and
  // that the creation is thread-safe.
  static std::vector<std::unique_ptr<GlobalMemoryResource>> mrs = _create_and_set_global_mrs();
}

}  // namespace legate::dataframe
