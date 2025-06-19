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

#include <unordered_map>

#include <legate.h>

#include <cudf/column/column_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace legate {
namespace dataframe {

class MemAlloc {
 public:
  MemAlloc() = default;
  MemAlloc(size_t nbytes, legate::Buffer<int8_t>&& buffer)
    : valid_{true}, nbytes_{nbytes}, buffer_{std::move(buffer)}
  {
  }
  MemAlloc(MemAlloc&& other) noexcept
    : valid_{std::exchange(other.valid_, false)},
      nbytes_{std::exchange(other.nbytes_, 0)},
      buffer_{std::move(other.buffer_)}
  {
  }

  MemAlloc& operator=(MemAlloc&& other)      = delete;
  MemAlloc(const MemAlloc& other)            = delete;
  MemAlloc& operator=(const MemAlloc& other) = delete;

  bool valid() const noexcept { return valid_; }
  size_t nbytes() const noexcept { return nbytes_; }
  legate::Buffer<int8_t>& buffer() noexcept { return buffer_; }

 private:
  bool valid_{false};
  size_t nbytes_{0};
  legate::Buffer<int8_t> buffer_;
};

/**
 * @brief Virtual base class for RMM device resources
 */
class DeviceMemoryResource : public rmm::mr::device_memory_resource {
 public:
  static inline constexpr size_t ALIGNMENT         = 16;
  static inline constexpr Memory::Kind MEMORY_KIND = legate::Memory::GPU_FB_MEM;
};

/**
 * @brief Task RMM resource that returns device memory allocations backed by legate buffers
 *
 * Only use this resource inside legate tasks e.g. as explicit arguments to libraries
 * such as libcudf. Do *not* set this resource as the default device resource through
 * RMM's `set_per_device_resource()` or `set_current_device_resource()`.
 *
 * Recommendation: let every task create and use a new instance of this class.
 */
class TaskMemoryResource : public DeviceMemoryResource {
 public:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override;

  /**
   * @brief Try to find the memory allocation of a base pointer
   *
   * If `base_ptr` points to the first element of an memory allocation, a pointer
   * to the allocation is returned otherwise `nullptr` is returned.
   *
   * @param base_ptr base memory pointer
   * @return Pointer to the memory allocation or `nullptr` if not found.
   */
  MemAlloc* find_buffer(const void* base_ptr);

  /**
   * @brief Return and release an allocated memory buffer
   *
   * If found, the buffer underlying `col` is released and returned. It will not be destroyed
   * at a later call to `.deallocate()`. Useful when a task moves a cudf column to its output.
   * If not found, an invalid buffer is returned.
   *
   * @param col cudf column view
   * @return A valid buffer if found
   */
  MemAlloc release_buffer(cudf::column_view col);

 private:
  std::unordered_map<const void*, MemAlloc> buffers_{};
};

/**
 * @brief Global RMM resource that return device memory allocations
 *
 * This resource returns allocations backed by either legate buffers or regular CUDA
 * buffers depending on whether the allocation is called from inside a legate task
 * or by the control code. Therefore, this resource can be used both inside and
 * outside tasks and can be used with RMM's `set_per_device_resource()` and
 * `set_current_device_resource()`.
 *
 * Recommendation: use `.set_as_default_mmr_resource()` to create a single instance of
 * this class per GPU and set them as the default memory resources.
 */
class GlobalMemoryResource : public DeviceMemoryResource {
 public:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override;

  /**
   * @brief Create an memory resource per GPU and set them as the default memory resources.
   *
   * This function is idempotent and only creates and sets the memory sources at first call.
   */
  static void set_as_default_mmr_resource();

 private:
  rmm::mr::cuda_memory_resource cuda_mr_{};
  TaskMemoryResource task_mr_{};
};

}  // namespace dataframe
}  // namespace legate
