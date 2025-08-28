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

#include <glob.h>
#include <stdlib.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "arrow/io/api.h"
#include <arrow/ipc/api.h>
#include <legate.h>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {
legate::Type to_legate_type(const arrow::DataType& arrow_type)
{
  switch (arrow_type.id()) {
    case arrow::Type::BOOL: {
      return legate::bool_();
    }
    case arrow::Type::INT8: {
      return legate::int8();
    }
    case arrow::Type::INT16: {
      return legate::int16();
    }
    case arrow::Type::INT32: {
      return legate::int32();
    }
    case arrow::Type::INT64: {
      return legate::int64();
    }
    case arrow::Type::UINT8: {
      return legate::uint8();
    }
    case arrow::Type::UINT16: {
      return legate::uint16();
    }
    case arrow::Type::UINT32: {
      return legate::uint32();
    }
    case arrow::Type::UINT64: {
      return legate::uint64();
    }
    case arrow::Type::FLOAT: {
      return legate::float32();
    }
    case arrow::Type::DOUBLE: {
      return legate::float64();
    }
    case arrow::Type::STRING: {
      return legate::string_type();
    }
    case arrow::Type::LARGE_STRING: {
      return legate::string_type();
    }
    case arrow::Type::DURATION: {
      return legate::int64();
    }
    case arrow::Type::DATE32: {
      return legate::int32();
    }
    case arrow::Type::DATE64: {
      return legate::int64();
    }
    case arrow::Type::TIMESTAMP: {
      return legate::int64();
    }
    case arrow::Type::TIME32: {
      return legate::int32();
    }
    case arrow::Type::TIME64: {
      return legate::int64();
    }
    default:
      throw std::invalid_argument("Converting arrow type to legate failed for type: " +
                                  arrow_type.ToString());
  }
}

std::shared_ptr<arrow::DataType> to_arrow_type(legate::Type::Code code)
{
  switch (code) {
    case legate::Type::Code::BOOL: {
      return arrow::boolean();
    }
    case legate::Type::Code::INT8: {
      return arrow::int8();
    }
    case legate::Type::Code::INT16: {
      return arrow::int16();
    }
    case legate::Type::Code::INT32: {
      return arrow::int32();
    }
    case legate::Type::Code::INT64: {
      return arrow::int64();
    }
    case legate::Type::Code::UINT8: {
      return arrow::uint8();
    }
    case legate::Type::Code::UINT16: {
      return arrow::uint16();
    }
    case legate::Type::Code::UINT32: {
      return arrow::uint32();
    }
    case legate::Type::Code::UINT64: {
      return arrow::uint64();
    }
    case legate::Type::Code::FLOAT32: {
      return arrow::float32();
    }
    case legate::Type::Code::FLOAT64: {
      return arrow::float64();
    }
    case legate::Type::Code::STRING: {
      return arrow::utf8();
    }
    default:
      throw std::invalid_argument("Unsupported Legate datatype: " +
                                  legate::primitive_type(code).to_string());
  }
}

namespace {

struct read_accessor_as_1d_bytes_fn {
  template <legate::Type::Code CODE>
  const void* operator()(const legate::PhysicalStore& store)
  {
    using VAL        = legate::type_of<CODE>;
    const auto shape = store.shape<1>();
    std::array<size_t, 1> strides{};
    const VAL* data = store.read_accessor<VAL, 1>().ptr(shape, strides.data());
    if (shape.volume() > 1 && strides[0] != 1) {
      throw std::runtime_error(
        "The store must be contiguous, please make sure the "
        "mapper sets `policy.exact = true`");
    }
    return reinterpret_cast<const void*>(data);
  }
};

}  // namespace

const void* read_accessor_as_1d_bytes(const legate::PhysicalStore& store)
{
  return legate::type_dispatch(store.code(), read_accessor_as_1d_bytes_fn{}, store);
}

namespace {
void _get_stores(const legate::PhysicalArray& ary, std::vector<legate::PhysicalStore>& ret)
{
  if (ary.nested()) {
    if (ary.type().code() == legate::Type::Code::STRING) {
      const legate::StringPhysicalArray a = ary.as_string_array();
      _get_stores(a.chars(), ret);
      _get_stores(a.ranges(), ret);
    } else {
      throw std::invalid_argument("nested dtype " + ary.type().to_string() + " isn't supported");
    }
  } else {
    ret.push_back(ary.data());
  }
  if (ary.nullable()) { ret.push_back(ary.null_mask()); }
}
}  // namespace

std::vector<legate::PhysicalStore> get_stores(const legate::PhysicalArray& ary)
{
  std::vector<legate::PhysicalStore> ret;
  _get_stores(ary, ret);
  return ret;
}

std::vector<std::string> parse_glob(const std::string& glob_string)
{
  std::vector<std::string> ret;
  glob_t pglob;
  int res = glob(glob_string.c_str(), GLOB_ERR | GLOB_TILDE | GLOB_MARK, NULL, &pglob);
  if (res != 0) {
    if (res == GLOB_NOMATCH) {
      throw std::invalid_argument("not files matches glob: \'" + glob_string + "\'");
    } else {
      throw std::runtime_error("Glob read error");
    }
  }
  try {
    ret.reserve(pglob.gl_pathc);
    for (size_t i = 0; i < pglob.gl_pathc; ++i) {
      const char* path = pglob.gl_pathv[i];
      ret.push_back(path);
    }
  } catch (...) {
    globfree(&pglob);
    throw;
  }
  globfree(&pglob);
  return ret;
}

TempDir::TempDir(const bool cleanup) : cleanup_{cleanup}
{
  std::string tpl{std::filesystem::temp_directory_path() / "legate-dataframe.XXXXXX"};
  if (mkdtemp(tpl.data()) == nullptr) {}
  dir_path_ = tpl;
}

TempDir::~TempDir() noexcept
{
  if (cleanup_) {
    try {
      std::filesystem::remove_all(dir_path_);
    } catch (...) {
      std::cout << "error while trying to remove " << dir_path_.string() << std::endl;
    }
  }
}

std::pair<size_t, size_t> evenly_partition_work(size_t work_size, size_t rank, size_t nranks)
{
  assert(rank < nranks);
  // Find the size of each rank (rounded down):
  size_t my_size  = work_size / nranks;
  size_t my_start = my_size * rank;
  // Adjust for the first n ranks reading one additional row each.
  size_t remaining_size = work_size - my_size * nranks;
  if (static_cast<size_t>(rank) < remaining_size) { my_size += 1; }
  my_start += std::min(remaining_size, static_cast<size_t>(rank));

  return std::make_pair(my_start, my_size);
}

namespace {
// Ported from Legate.core:
// <https://github.com/nv-legate/legate.core/blob/0f509a00/src/core/utilities/linearize.cc#L22-L39>
struct linearize_fn {
  template <int DIM>
  [[nodiscard]] size_t operator()(const legate::DomainPoint& lo_dp,
                                  const legate::DomainPoint& hi_dp,
                                  const legate::DomainPoint& point_dp) const
  {
    const legate::Point<DIM> lo      = lo_dp;
    const legate::Point<DIM> hi      = hi_dp;
    const legate::Point<DIM> point   = point_dp;
    const legate::Point<DIM> extents = hi - lo + legate::Point<DIM>::ONES();
    size_t idx{0};
    for (int32_t dim = 0; dim < DIM; ++dim) {
      idx = idx * extents[dim] + point[dim] - lo[dim];
    }
    return idx;
  }
};
}  // namespace

size_t linearize(const legate::DomainPoint& lo,
                 const legate::DomainPoint& hi,
                 const legate::DomainPoint& point)
{
  return legate::dim_dispatch(point.dim, linearize_fn{}, lo, hi, point);
}

std::vector<uint8_t> serialize_arrow_type(std::shared_ptr<arrow::DataType> type)
{
  auto schema = arrow::schema({arrow::field("type", type)});
  auto buffer = ARROW_RESULT(arrow::ipc::SerializeSchema(*schema));
  return std::vector<uint8_t>(buffer->data(), buffer->data() + buffer->size());
}

std::vector<uint8_t> serialize_arrow_types(std::vector<std::shared_ptr<arrow::DataType>> types)
{
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto& type : types) {
    fields.push_back(arrow::field(std::to_string(fields.size()), type));
  }
  auto schema = arrow::schema(fields);
  auto buffer = ARROW_RESULT(arrow::ipc::SerializeSchema(*schema));
  return std::vector<uint8_t>(buffer->data(), buffer->data() + buffer->size());
}

std::shared_ptr<arrow::DataType> deserialize_arrow_type(const std::vector<uint8_t>& data)
{
  auto buffer       = arrow::Buffer::Wrap(data.data(), data.size());
  auto input_stream = arrow::io::BufferReader(buffer);
  auto schema       = ARROW_RESULT(arrow::ipc::ReadSchema(&input_stream, nullptr));
  return schema->fields().at(0)->type();
}

std::vector<std::shared_ptr<arrow::DataType>> deserialize_arrow_types(
  const std::vector<uint8_t>& data)
{
  auto buffer       = arrow::Buffer::Wrap(data.data(), data.size());
  auto input_stream = arrow::io::BufferReader(buffer);
  auto schema       = ARROW_RESULT(arrow::ipc::ReadSchema(&input_stream, nullptr));
  std::vector<std::shared_ptr<arrow::DataType>> types;
  for (const auto& field : schema->fields()) {
    types.push_back(field->type());
  }
  return types;
}
}  // namespace legate::dataframe
