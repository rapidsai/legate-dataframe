/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <vector>

#include <legate_dataframe/core/table.hpp>

namespace legate::dataframe {

/**
 * @brief Hash partition a table based on some columns.
 *
 * This returns a new table and partitioning information
 *
 * TODO: Should maybe just attach that to the table.  Need a convenient way
 * (or at least an example) for applying the partitioning constraints.
 *
 * If we attach some user unique "partition" object (maybe?) then the
 * user could drag that around.
 * I.e. a partition is described by the column (names) a second step working
 * on a partitioned table will always have to use those identical names in some way
 * (e.g. for a local join).  (in fact, maybe to the point that remembering the
 * hashes might be useful often?)
 *
 * @param tbl The table to hash partition.
 * @param keys The column names to partition by.
 * @param num_parts Number of partitions or -1 in which case the current number
 * of ranks will be used.  (TODO: To be seen what makes sense here.)
 * @return A new LogicalTable with and a LogicalArray describing its partitioning.
 */
std::pair<LogicalTable, legate::LogicalArray>
hashpartition(const LogicalTable& tbl, const std::set<std::string>& keys, int num_parts = -1);


// TODO: Do we really duplicate docs for this?
std::pair<LogicalTable, legate::LogicalArray>
hashpartition(const LogicalTable& tbl, const std::set<size_t>& keys, int num_parts = -1);


}  // namespace legate::dataframe
