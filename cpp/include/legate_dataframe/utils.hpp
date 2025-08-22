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

#include <filesystem>

#include <arrow/api.h>
#include <legate.h>

#include <arrow/type.h>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

namespace legate::dataframe {

[[nodiscard]] cudf::type_id to_cudf_type_id(legate::Type::Code code);
[[nodiscard]] std::shared_ptr<arrow::DataType> to_arrow_type(cudf::type_id code);
[[nodiscard]] std::shared_ptr<arrow::DataType> to_arrow_type(legate::Type::Code code);
[[nodiscard]] cudf::data_type to_cudf_type(const arrow::DataType& arrow_type);
[[nodiscard]] inline cudf::data_type to_cudf_type(
  const std::shared_ptr<arrow::DataType>& arrow_type)
{
  return to_cudf_type(*arrow_type);
}
[[nodiscard]] legate::Type to_legate_type(cudf::type_id dtype);
[[nodiscard]] legate::Type to_legate_type(const arrow::DataType& arrow_type);

std::string pprint_1d(cudf::column_view col,
                      cudf::size_type index,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr);

const void* read_accessor_as_1d_bytes(const legate::PhysicalStore& store);

std::vector<legate::PhysicalStore> get_stores(const legate::PhysicalArray& ary);

/**
 * @brief Helper unpack an arrow result or throw an exception
 *
 * @tparam T The type of the result
 * @param result The Arrow result to convert
 * @return The result
 */
template <typename T>
T ARROW_RESULT(arrow::Result<T> result)
{
  if (!result.ok()) { throw std::runtime_error(result.status().ToString()); }
  return std::move(result).ValueOrDie();
}

/**
 * @brief Parse a UNIX glob string incl. tilde expansion
 *
 * Glob syntax: <https://linux.die.net/man/7/glob>
 *
 * @param glob_string The glob string to parse
 * @return The glob matching paths (both directories and files)
 */
std::vector<std::string> parse_glob(const std::string& glob_string);

class TempDir {
 public:
  TempDir(const bool cleanup_ = true);
  ~TempDir() noexcept;

  const std::filesystem::path& path() { return dir_path_; }

  operator std::string() { return path(); }

 private:
  const bool cleanup_;
  std::filesystem::path dir_path_;
};

/**
 * @brief Indicates whether the memory kind is device memory or not
 *
 * @param mem_kind The memory kind to query
 * @return true  The memory kind is device memory
 * @return false The memory kind is not device memory
 */
[[nodiscard]] inline bool is_device_mem(legate::Memory::Kind mem_kind)
{
  return mem_kind == legate::Memory::Kind::GPU_DYNAMIC_MEM ||
         mem_kind == legate::Memory::Kind::GPU_FB_MEM;
}

/**
 * @brief Concatenate two containers by inserting into the primary one
 *
 * @tparam PrimaryContainer The type of the primary container
 * @tparam OtherContainer The type of the other container
 * @param primary The primary collection, which will be copied
 * @param other The other collection, which will be inserted into @p primary
 * @return A copy of @p primary where each element in @p other are inserted
 */
template <typename PrimaryContainer, typename OtherContainer>
[[nodiscard]] PrimaryContainer concat(PrimaryContainer primary, const OtherContainer& other)
{
  primary.insert(primary.end(), other.begin(), other.end());
  return primary;
}

/**
 * @brief Concatenate two containers by inserting into the primary one
 *
 * @tparam PrimaryContainer The type of the primary container
 * @tparam OtherContainer The type of the other container
 * @param primary The primary collection, which will be copied
 * @param other The other collection, which will be inserted into @p primary
 * @return A copy of @p primary where each element in @p other are inserted
 */
template <typename PrimaryContainer, typename OtherContainer>
[[nodiscard]] PrimaryContainer concat(PrimaryContainer primary, OtherContainer&& other)
{
  primary.insert(
    primary.end(), std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
  return primary;
}

/**
 * @brief Helper to get a partition of the full work size assigned.
 *
 * In some places, the inputs are not columns and the work is not implicitly
 * divided.  In this case, you can use `evenly_partition_work` to find which
 * part of the total work this rank should perform.
 * The `work_size` will be split evenly, with the first ranks being assigned
 * one additional item if necessary.
 *
 * @param work_size The total work size to be split between all ranks.
 * @param rank The partition to return.
 * @param nranks The total number of partitions to split into.
 * @returns The start and size of the assigned work.
 */
[[nodiscard]] std::pair<size_t, size_t> evenly_partition_work(size_t work_size,
                                                              size_t rank,
                                                              size_t nranks);

/**
 * @brief Flatten a legate domain point to an integer
 *
 * @param lo Lower point
 * @param hi High point
 * @param point The domain point to flatten
 * @return The flattened domain point
 */
size_t linearize(const legate::DomainPoint& lo,
                 const legate::DomainPoint& hi,
                 const legate::DomainPoint& point);

/**
 * @brief Serialize an Arrow data type to a byte vector
 *
 * @param type The Arrow data type to serialize
 * @return A byte vector containing the serialized data
 */
std::vector<uint8_t> serialize_arrow_type(std::shared_ptr<arrow::DataType> type);

/*
 * @brief Serialize a vector of Arrow data types to a byte vector
 *
 * @param types The vector of Arrow data types to serialize
 * @return A byte vector containing the serialized data
 */
std::vector<uint8_t> serialize_arrow_types(std::vector<std::shared_ptr<arrow::DataType>> types);

/**
 * @brief Deserialize a byte vector into an Arrow data type
 *
 * @param data The byte vector containing the serialized data
 * @return The deserialized Arrow data type
 */
std::shared_ptr<arrow::DataType> deserialize_arrow_type(const std::vector<uint8_t>& data);

/*
 * @brief Deserialize a byte vector into a vector of Arrow data types
 *
 * @param data The byte vector containing the serialized data
 * @return The deserialized vector of Arrow data types
 */
std::vector<std::shared_ptr<arrow::DataType>> deserialize_arrow_types(
  const std::vector<uint8_t>& data);

/*
 * @brief If a store is unbound, bind a buffer to it of size and return its pointer, if already
 * bound, then check the buffer is of size and return its pointer.
 *
 * @param store The physical store to bind the buffer to
 * @param size The size of the buffer
 * @return A pointer to the buffer.
 */
template <typename T>
T* maybe_bind_buffer(legate::PhysicalStore store, std::size_t size)
{
  T* out;
  if (store.is_unbound_store()) {
    out = store.create_output_buffer<T, 1>(legate::Point<1>(size), true).ptr(0);
  } else {
    auto acc = store.write_accessor<T, 1>();
    assert((store.shape<1>().hi[0] - store.shape<1>().lo[0]) == -1 ||
           acc.accessor.is_dense_row_major(store.shape<1>()));
    if (store.shape<1>().volume() != size) {
      throw std::runtime_error("Store size does not match the expected size");
    }
    out = acc.ptr(store.shape<1>().lo[0]);
  }
  return out;
}

}  // namespace legate::dataframe
