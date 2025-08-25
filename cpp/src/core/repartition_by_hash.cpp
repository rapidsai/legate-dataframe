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

#include <sstream>
#include <stdexcept>

#include "legate/comm/coll.h"
#include <legate.h>
#include <legate_dataframe/core/library.hpp>

#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include <arrow/acero/api.h>
#include <arrow/compute/api.h>

#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

namespace {
// Sendcounts contains the size of the buffer send to the j-th rank.
// Returns recvcounts: the size of the buffer received from the j-th rank.
std::vector<int> get_recvcounts(TaskContext& ctx, const std::vector<int>& sendcounts)
{
  std::vector<int> recvcounts(ctx.nranks);
  legate::comm::coll::collAlltoall(
    sendcounts.data(),
    recvcounts.data(),
    1,
    legate::comm::coll::CollDataType::CollInt,
    ctx.get_legate_context().communicator(0).get<legate::comm::coll::CollComm>());

  return recvcounts;
}

}  // namespace

/**
 * @brief Shuffle (all-to-all exchange) packed arrow partitioned table.
 *
 *
 * @param ctx The context of the calling task
 * @param tbl_partitioned The local table partitioned into multiple tables such
 * that `tbl_partitioned.at(i)` should end up at rank i.
 * @return A vector of tables sent to this rank by other ranks.
 */
std::vector<std::shared_ptr<arrow::Table>> shuffle(
  TaskContext& ctx, const std::vector<std::shared_ptr<arrow::Table>>& tbl_partitioned)
{
  if (tbl_partitioned.size() != ctx.nranks) {
    throw std::runtime_error("internal error: partition split has wrong size.");
  }
  for (auto& tbl : tbl_partitioned) {
    if (!tbl) { throw std::runtime_error("internal error: partitioned table is null."); }
  }

  if (ctx.get_legate_context().communicators().empty()) {
    throw std::runtime_error("internal error: communicator not initialized.");
  }
  auto comm      = ctx.get_legate_context().communicator(0);
  auto* comm_ptr = comm.get<legate::comm::coll::CollComm>();

  // Serialize table as buffers
  std::vector<std::shared_ptr<arrow::Buffer>> send_buffers;
  std::vector<int> sendcounts;
  for (const auto& tbl : tbl_partitioned) {
    auto output_stream = ARROW_RESULT(arrow::io::BufferOutputStream::Create());
    auto writer        = ARROW_RESULT(arrow::ipc::MakeStreamWriter(output_stream, tbl->schema()));

    auto status_written = writer->WriteTable(*tbl);
    auto status_closed  = writer->Close();
    if (!status_written.ok() || !status_closed.ok()) {
      auto status = status_written.ok() ? status_closed : status_written;
      std::stringstream ss;
      ss << "Failed to write table to stream: " << status.ToString();
      throw std::runtime_error(ss.str());
    }
    send_buffers.push_back(ARROW_RESULT(output_stream->Finish()));
    sendcounts.push_back(send_buffers.back()->size());
  }

  std::size_t total_send_size =
    std::accumulate(sendcounts.begin(), sendcounts.end(), 0, std::plus<std::size_t>());
  auto sendbuffer = legate::create_buffer<uint8_t>(total_send_size);

  // Pack the buffers into a single buffer
  std::size_t offset = 0;
  std::vector<int> displacements_send;
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    displacements_send.push_back(offset);
    auto& buf = send_buffers[i];
    std::memcpy(sendbuffer.ptr(offset), buf->data(), buf->size());
    offset += buf->size();
  }

  // Communicate the sizes to all ranks
  auto recvcounts = get_recvcounts(ctx, sendcounts);
  std::vector<int> displacements_recv;
  std::size_t total_recv_size = 0;
  for (size_t i = 0; i < recvcounts.size(); ++i) {
    displacements_recv.push_back(total_recv_size);
    total_recv_size += recvcounts[i];
  }

  // Use an arrow buffer here instead of a legate buffer
  // Arrow will propagate shared pointers ensuring the buffer is valid
  std::shared_ptr<arrow::Buffer> recvbuffer = ARROW_RESULT(arrow::AllocateBuffer(total_recv_size));
  comm::coll::collAlltoallv(sendbuffer.ptr(0),
                            sendcounts.data(),
                            displacements_send.data(),
                            recvbuffer->mutable_data(),
                            recvcounts.data(),
                            displacements_recv.data(),
                            comm::coll::CollDataType::CollInt8,
                            comm_ptr);

  std::vector<std::shared_ptr<arrow::Table>> result;
  offset = 0;
  for (size_t i = 0; i < recvcounts.size(); ++i) {
    auto buffer = ARROW_RESULT(arrow::SliceBufferSafe(recvbuffer, offset, recvcounts[i]));
    offset += recvcounts[i];
    auto input_stream = arrow::io::BufferReader(buffer);
    auto reader       = ARROW_RESULT(arrow::ipc::RecordBatchStreamReader::Open(&input_stream));
    result.push_back(ARROW_RESULT(reader->ToTable()));
  }

  sendbuffer.destroy();
  return result;
}

// Boost hash combine
inline int64_t hash_combine(int64_t seed, const int64_t& v)
{
  std::hash<int64_t> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

// Partition the Arrow table by hashing key columns into n_ranks bins
// Result is a map of key to table
std::vector<std::shared_ptr<arrow::Table>> partition_arrow_table(
  TaskContext& ctx,
  std::shared_ptr<arrow::Table> table,
  const std::vector<std::size_t>& columns_to_hash)
{
  // Assign unique keys to ranks based on hash function
  std::vector<uint64_t> key_hashes(table->num_rows());
  for (int64_t i = 0; i < table->num_rows(); ++i) {
    uint64_t hash = 0;
    for (const auto& col_idx : columns_to_hash) {
      auto col = table->column(col_idx);
      hash     = hash_combine(ARROW_RESULT(col->GetScalar(i))->hash(), hash);
    }
    key_hashes.at(i) = hash;
  }
  for (auto& hash : key_hashes) {
    hash = hash % ctx.nranks;
  }
  arrow::UInt64Builder builder;
  auto status           = builder.AppendValues(key_hashes);
  auto key_hash_col     = ARROW_RESULT(builder.Finish());
  auto hash_column_name = "destination_rank";
  auto table_with_hash =
    ARROW_RESULT(table->AddColumn(table->num_columns(),
                                  arrow::field(hash_column_name, key_hash_col->type()),
                                  std::make_shared<arrow::ChunkedArray>(key_hash_col)));

  std::vector<arrow::compute::Aggregate> aggregations;
  for (auto column_name : table->ColumnNames()) {
    aggregations.emplace_back("hash_list", column_name, column_name);
  }
  arrow::acero::Declaration plan = arrow::acero::Declaration::Sequence(
    {{"table_source", arrow::acero::TableSourceNodeOptions(table_with_hash)},
     {"aggregate", arrow::acero::AggregateNodeOptions(aggregations, {hash_column_name})}});
  // Each row of the 'lists' table contains a table for the group
  auto lists = ARROW_RESULT(arrow::acero::DeclarationToTable(std::move(plan)));
  auto keys  = lists->GetColumnByName(hash_column_name);

  // Fill with empty table in case no keys for a rank
  std::vector<std::shared_ptr<arrow::Table>> result(
    ctx.nranks, ARROW_RESULT(arrow::Table::MakeEmpty(table->schema())));
  for (int i = 0; i < lists->num_rows(); ++i) {
    auto key =
      std::dynamic_pointer_cast<arrow::UInt64Scalar>(ARROW_RESULT(keys->GetScalar(i)))->value;
    std::vector<std::shared_ptr<arrow::Array>> partition_columns;
    for (auto name : table->ColumnNames()) {
      auto list = lists->GetColumnByName(name);
      auto list_scalar =
        std::dynamic_pointer_cast<arrow::ListScalar>(ARROW_RESULT(list->GetScalar(i)));
      partition_columns.push_back(list_scalar->value);
    }
    result.at(key) = arrow::Table::Make(table->schema(), partition_columns);
  }

  return result;
}

std::shared_ptr<arrow::Table> repartition_by_hash(TaskContext& ctx,
                                                  std::shared_ptr<arrow::Table> table,
                                                  const std::vector<std::size_t>& columns_to_hash)
{
  if (ctx.nranks == 1) { return table; }

  std::vector<std::shared_ptr<arrow::Table>> partitioned_table(ctx.nranks);
  if (table->num_rows() == 0) {
    for (auto& p : partitioned_table) {
      p = ARROW_RESULT(arrow::Table::MakeEmpty(table->schema()));
    }
  } else {
    partitioned_table = partition_arrow_table(ctx, table, columns_to_hash);
  }

  for (const auto& tbl : partitioned_table) {
    auto status = tbl->ValidateFull();
    if (!status.ok()) {
      std::stringstream ss;
      ss << "Failed to validate table after repartitioning: " << status.ToString();
      throw std::runtime_error(ss.str());
    }
  }
  auto tables = shuffle(ctx, partitioned_table);
  // Validate the tables
  for (const auto& tbl : tables) {
    auto status = tbl->ValidateFull();
    if (!status.ok()) {
      std::stringstream ss;
      ss << "Failed to validate table after repartitioning: " << status.ToString();
      throw std::runtime_error(ss.str());
    }
  }
  return ARROW_RESULT(arrow::ConcatenateTables(tables));
}
}  // namespace legate::dataframe::task
