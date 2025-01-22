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

#include <string>

#include <cudf/datetime.hpp>
#include <cudf/types.hpp>

#include <legate_dataframe/core/column.hpp>

namespace legate::dataframe {

/**
 * @brief Returns a new timestamp column converting a strings column into
 * timestamps using the provided format pattern.
 *
 * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z"
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | \%d | Day of the month: 01-31 |
 * | \%m | Month of the year: 01-12 |
 * | \%y | Year without century: 00-99. [0,68] maps to [2000,2068] and [69,99] maps to [1969,1999] |
 * | \%Y | Year with century: 0001-9999 |
 * | \%H | 24-hour of the day: 00-23 |
 * | \%I | 12-hour of the day: 01-12 |
 * | \%M | Minute of the hour: 00-59 |
 * | \%S | Second of the minute: 00-59. Leap second is not supported. |
 * | \%f | 6-digit microsecond: 000000-999999 |
 * | \%z | UTC offset with format ±HHMM Example +0500 |
 * | \%j | Day of the year: 001-366 |
 * | \%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
 * | \%W | Week of the year with Monday as the first day of the week: 00-53 |
 * | \%w | Day of week: 0-6 = Sunday-Saturday |
 * | \%U | Week of the year with Sunday as the first day of the week: 00-53 |
 * | \%u | Day of week: 1-7 = Monday-Sunday |
 *
 * Other specifiers are not currently supported.
 *
 * Invalid formats are not checked. If the string contains unexpected
 * or insufficient characters, that output row entry's timestamp value is undefined.
 *
 * Any null string entry will result in a corresponding null row in the output column.
 *
 * The resulting time units are specified by the `timestamp_type` parameter.
 * The time units are independent of the number of digits parsed by the "%f" specifier.
 * The "%f" supports a precision value to read the numeric digits. Specify the
 * precision with a single integer value (1-9) as follows:
 * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
 *
 * Although leap second is not supported for "%S", no checking is performed on the value.
 * The cudf::strings::is_timestamp can be used to verify the valid range of values.
 *
 * If "%W"/"%w" (or "%U/%u") and "%m"/"%d" are both specified, the "%W"/%U and "%w"/%u values
 * take precedent when computing the date part of the timestamp result.
 *
 * @throw cudf::logic_error if timestamp_type is not a timestamp type.
 *
 * @param input Strings instance for this operation
 * @param timestamp_type The timestamp type used for creating the output column
 * @param format String specifying the timestamp format in strings
 * @return New datetime column
 */
LogicalColumn to_timestamps(const LogicalColumn& input,
                            cudf::data_type timestamp_type,
                            std::string format);

/**
 * @brief Extracts part of a timestamp as a int16.
 *
 * @param input Timestamp column
 * @param component The component which to extract.
 * @return New int16 column.
 */
LogicalColumn extract_timestamp_component(const LogicalColumn& input,
                                          cudf::datetime::datetime_component component);

}  // namespace legate::dataframe
