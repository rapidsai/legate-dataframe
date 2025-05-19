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

#include <cudf/copying.hpp>

#include <legate.h>
#include <legate/cuda/stream_pool.h>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

namespace {
struct pprint_1d_scalar_fn {
  template <typename T, std::enable_if_t<cudf::is_numeric<T>()>* = nullptr>
  std::string operator()(cudf::column_view col,
                         cudf::size_type index,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  {
    std::unique_ptr<cudf::scalar> scalar = cudf::get_element(col, index, stream, mr);
    auto typed_scalar = static_cast<cudf::numeric_scalar<T> const*>(scalar.get());
    T val             = typed_scalar->value(stream);
    return std::to_string(val);
  }
  template <typename T, std::enable_if_t<!cudf::is_numeric<T>()>* = nullptr>
  std::string operator()(cudf::column_view col,
                         cudf::size_type index,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr)
  {
    throw std::runtime_error("not implemented");
  }
};
}  // namespace

std::string pprint_1d(cudf::column_view col,
                      cudf::size_type index,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(col.type(), pprint_1d_scalar_fn{}, col, index, stream, mr);
}

cudf::type_id to_cudf_type_id(legate::Type::Code code)
{
  switch (code) {
    case legate::Type::Code::BOOL: {
      return cudf::type_id::BOOL8;
    }
    case legate::Type::Code::INT8: {
      return cudf::type_id::INT8;
    }
    case legate::Type::Code::INT16: {
      return cudf::type_id::INT16;
    }
    case legate::Type::Code::INT32: {
      return cudf::type_id::INT32;
    }
    case legate::Type::Code::INT64: {
      return cudf::type_id::INT64;
    }
    case legate::Type::Code::UINT8: {
      return cudf::type_id::UINT8;
    }
    case legate::Type::Code::UINT16: {
      return cudf::type_id::UINT16;
    }
    case legate::Type::Code::UINT32: {
      return cudf::type_id::UINT32;
    }
    case legate::Type::Code::UINT64: {
      return cudf::type_id::UINT64;
    }
    case legate::Type::Code::FLOAT32: {
      return cudf::type_id::FLOAT32;
    }
    case legate::Type::Code::FLOAT64: {
      return cudf::type_id::FLOAT64;
    }
    case legate::Type::Code::STRING: {
      return cudf::type_id::STRING;
    }
    default:
      throw std::runtime_error("Unsupported Legate datatype: " +
                               legate::primitive_type(code).to_string());
  }
}

legate::Type to_legate_type(cudf::type_id dtype)
{
  switch (dtype) {
    case cudf::type_id::INT8: {
      return legate::int8();
    }
    case cudf::type_id::INT16: {
      return legate::int16();
    }
    case cudf::type_id::INT32: {
      return legate::int32();
    }
    case cudf::type_id::INT64: {
      return legate::int64();
    }
    case cudf::type_id::UINT8: {
      return legate::uint8();
    }
    case cudf::type_id::UINT16: {
      return legate::uint16();
    }
    case cudf::type_id::UINT32: {
      return legate::uint32();
    }
    case cudf::type_id::UINT64: {
      return legate::uint64();
    }
    case cudf::type_id::FLOAT32: {
      return legate::float32();
    }
    case cudf::type_id::FLOAT64: {
      return legate::float64();
    }
    case cudf::type_id::BOOL8: {
      return legate::bool_();
    }
    case cudf::type_id::STRING: {
      return legate::string_type();
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      return legate::int32();
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      return legate::int64();
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      return legate::int64();
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      return legate::int64();
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      return legate::int64();
    }
    case cudf::type_id::DURATION_DAYS: {
      return legate::int32();
    }
    case cudf::type_id::DURATION_SECONDS: {
      return legate::int64();
    }
    case cudf::type_id::DURATION_MILLISECONDS: {
      return legate::int64();
    }
    case cudf::type_id::DURATION_MICROSECONDS: {
      return legate::int64();
    }
    case cudf::type_id::DURATION_NANOSECONDS: {
      return legate::int64();
    }
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    default:
      throw std::invalid_argument(
        "unsupported cudf datatype: " +
        std::to_string(static_cast<std::underlying_type_t<cudf::type_id>>(dtype)));
  }
}

std::shared_ptr<arrow::DataType> to_arrow_type(cudf::type_id code)
{
  switch (code) {
    case cudf::type_id::INT8: {
      return arrow::int8();
    }
    case cudf::type_id::INT16: {
      return arrow::int16();
    }
    case cudf::type_id::INT32: {
      return arrow::int32();
    }
    case cudf::type_id::INT64: {
      return arrow::int64();
    }
    case cudf::type_id::UINT8: {
      return arrow::uint8();
    }
    case cudf::type_id::UINT16: {
      return arrow::uint16();
    }
    case cudf::type_id::UINT32: {
      return arrow::uint32();
    }
    case cudf::type_id::UINT64: {
      return arrow::uint64();
    }
    case cudf::type_id::FLOAT32: {
      return arrow::float32();
    }
    case cudf::type_id::FLOAT64: {
      return arrow::float64();
    }
    // Don't use arrow::boolean() - arrow::BooleanArray uses bitpacking
    case cudf::type_id::BOOL8: {
      return arrow::uint8();
    }
    case cudf::type_id::STRING: {
      return arrow::utf8();
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      return arrow::timestamp(arrow::TimeUnit::SECOND);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      return arrow::timestamp(arrow::TimeUnit::MILLI);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      return arrow::timestamp(arrow::TimeUnit::MICRO);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      return arrow::timestamp(arrow::TimeUnit::NANO);
    }
    case cudf::type_id::DURATION_SECONDS: {
      return arrow::duration(arrow::TimeUnit::SECOND);
    }
    case cudf::type_id::DURATION_MILLISECONDS: {
      return arrow::duration(arrow::TimeUnit::MILLI);
    }
    case cudf::type_id::DURATION_MICROSECONDS: {
      return arrow::duration(arrow::TimeUnit::MICRO);
    }
    case cudf::type_id::DURATION_NANOSECONDS: {
      return arrow::duration(arrow::TimeUnit::NANO);
    }
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    default:
      throw std::invalid_argument(
        "unsupported cudf datatype: " +
        std::to_string(static_cast<std::underlying_type_t<cudf::type_id>>(code)));
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

bool operator==(legate::PhysicalArray array, legate::PhysicalArray other_array)
{
  if (array.nullable() != other_array.nullable()) return false;
  if (array.type() != other_array.type()) return false;
  if (array.dim() != other_array.dim()) return false;
  if (array.shape<1>() != other_array.shape<1>()) return false;
  auto data       = array.data().template read_accessor<uint8_t, 1, false>().ptr(0);
  auto type       = array.type();
  auto other_data = other_array.data().template read_accessor<uint8_t, 1>().ptr(0);
  auto shape      = array.shape<1>();
  // If element is null, data need not be the same
  if (array.nullable()) {
    auto null_accessor       = array.null_mask().template read_accessor<bool, 1>();
    auto other_null_accessor = other_array.null_mask().template read_accessor<bool, 1>();
    for (size_t i = 0; i < shape.volume(); ++i) {
      if (null_accessor[i] != other_null_accessor[i]) return false;
      // Element exists, check data is the same
      if (null_accessor[i]) {
        for (size_t j = 0; j < type.size(); ++j) {
          if (data[i * type.size() + j] != other_data[i * type.size() + j]) return false;
        }
      }
    }
  } else {
    // No mask, expect data to be bitwise identical
    for (size_t i = 0; i < shape.volume() * type.size(); ++i) {
      if (data[i] != other_data[i]) return false;
    }
  }
  return true;
}
}  // namespace legate::dataframe
