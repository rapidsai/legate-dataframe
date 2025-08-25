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
      throw std::invalid_argument("Unsupported Legate datatype: " +
                                  legate::primitive_type(code).to_string());
  }
}

cudf::data_type to_cudf_type(const arrow::DataType& arrow_type)
{
  switch (arrow_type.id()) {
    case arrow::Type::BOOL: {
      return cudf::data_type{cudf::type_id::BOOL8};
    }
    case arrow::Type::INT8: {
      return cudf::data_type{cudf::type_id::INT8};
    }
    case arrow::Type::INT16: {
      return cudf::data_type{cudf::type_id::INT16};
    }
    case arrow::Type::INT32: {
      return cudf::data_type{cudf::type_id::INT32};
    }
    case arrow::Type::INT64: {
      return cudf::data_type{cudf::type_id::INT64};
    }
    case arrow::Type::UINT8: {
      return cudf::data_type{cudf::type_id::UINT8};
    }
    case arrow::Type::UINT16: {
      return cudf::data_type{cudf::type_id::UINT16};
    }
    case arrow::Type::UINT32: {
      return cudf::data_type{cudf::type_id::UINT32};
    }
    case arrow::Type::UINT64: {
      return cudf::data_type{cudf::type_id::UINT64};
    }
    case arrow::Type::FLOAT: {
      return cudf::data_type{cudf::type_id::FLOAT32};
    }
    case arrow::Type::DOUBLE: {
      return cudf::data_type{cudf::type_id::FLOAT64};
    }
    case arrow::Type::STRING: {
      return cudf::data_type{cudf::type_id::STRING};
    }
    case arrow::Type::LARGE_STRING: {
      return cudf::data_type{cudf::type_id::STRING};
    }
    case arrow::Type::DATE64: {
      return cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
    }
    case arrow::Type::DURATION: {
      const auto& duration_type = static_cast<const arrow::DurationType&>(arrow_type);
      if (duration_type.unit() == arrow::TimeUnit::SECOND) {
        return cudf::data_type{cudf::type_id::DURATION_SECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::MILLI) {
        return cudf::data_type{cudf::type_id::DURATION_MILLISECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::MICRO) {
        return cudf::data_type{cudf::type_id::DURATION_MICROSECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::NANO) {
        return cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
      }
      break;
    }
    case arrow::Type::DATE32: {
      return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    }
    case arrow::Type::TIMESTAMP: {
      const auto& duration_type = static_cast<const arrow::DurationType&>(arrow_type);
      if (duration_type.unit() == arrow::TimeUnit::SECOND) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::MILLI) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::MICRO) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
      } else if (duration_type.unit() == arrow::TimeUnit::NANO) {
        return cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
      }
      break;
    }
    default: break;
  }
  throw std::invalid_argument("Converting arrow type to cudf failed for type: " +
                              arrow_type.ToString());
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

}  // namespace legate::dataframe
