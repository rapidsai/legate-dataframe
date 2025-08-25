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

#include <cuda_runtime_api.h>

#include "legate/comm/coll.h"
#include <legate.h>
#include <legate/cuda/cuda.h>
#include <legate_dataframe/core/library.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/contiguous_split.hpp>  // `cudf::detail::pack`
#include <cudf/partitioning.hpp>

#include <legate_dataframe/core/nccl.hpp>
#include <legate_dataframe/core/repartition_by_hash.hpp>
#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

namespace {

/**
 * @brief Help class for exchanging buffer sizes of the packed cudf columns
 */
class ExchangedSizes {
 private:
  legate::Buffer<std::size_t> _all_sizes;
  TaskContext& _ctx;

 public:
  // We use a temporary stream for the metadata communication. This ways, we avoid
  // synchronizing the main task stream.
  cudaStream_t stream;

  /**
   * @brief Exchange (all-to-all) the sizes of the packed cudf columns.
   *
   * When constructed, use `metadata` and `gpu_data` to get the buffer size for a specific rank.
   *
   * @param ctx The context of the calling task
   * @param columns A mapping of tasks to their packed columns. E.g. `columns.at(i)`
   * will be send to the i'th task. NB: all tasks beside itself must have a map thus:
   * `columns.size() == ctx.nranks - 1`.
   */
  ExchangedSizes(TaskContext& ctx, const std::map<int, cudf::packed_columns>& columns) : _ctx(ctx)
  {
    LEGATE_CHECK_CUDA(cudaStreamCreate(&stream));
    assert(columns.size() == ctx.nranks - 1);
    // Note: Size of this buffer is taken into account in the mapper:
    _all_sizes =
      legate::create_buffer<std::size_t>(ctx.nranks * ctx.nranks * 2, Memory::Kind::Z_COPY_MEM);

    // Copy the sizes of the metadata and gpu_data of each packed column into _all_sizes at the
    // location corresponding to our rank.
    const int stride = ctx.nranks * 2;
    for (int i = 0; i < ctx.nranks; ++i) {
      if (i == ctx.rank) {
        _all_sizes[ctx.rank * stride + i]              = 0;
        _all_sizes[ctx.rank * stride + ctx.nranks + i] = 0;
      } else {
        _all_sizes[ctx.rank * stride + i]              = columns.at(i).metadata->size();
        _all_sizes[ctx.rank * stride + ctx.nranks + i] = columns.at(i).gpu_data->size();
      }
    }

    // We have to sync here before proceeding as we need the sizes to arrive in order to
    // allocate communication buffers later.
    auto task = ctx.get_legate_context();
    task.concurrent_task_barrier();
    CHECK_NCCL(ncclAllGather(_all_sizes.ptr(ctx.rank * stride),
                             _all_sizes.ptr(0),
                             stride,
                             ncclUint64,
                             task_nccl(ctx),
                             stream));
    LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream));
    task.concurrent_task_barrier();
  }

  ~ExchangedSizes() { LEGATE_CHECK_CUDA(cudaStreamDestroy(stream)); }

  // TODO: implement a destructor that syncs and calls _all_sizes.destroy(). Currently,
  //       the lifespan of `_all_sizes` is until the legate task finish.

  /**
   * @brief Get the size of the metadata send between two ranks.
   *
   * @param src_rank The rank of the sending task
   * @param dst_rank The rank of the receiving task
   * @return Size of the metadata buffer (in bytes)
   */
  std::size_t metadata(int src_rank, int dst_rank)
  {
    return _all_sizes[src_rank * _ctx.nranks * 2 + dst_rank];
  }
  /**
   * @brief Get the size of the GPU data send between two ranks.
   *
   * @param src_rank The rank of the sending task
   * @param dst_rank The rank of the receiving task
   * @return Size of the GPU device buffer (in bytes)
   */
  std::size_t gpu_data(int src_rank, int dst_rank)
  {
    return _all_sizes[src_rank * _ctx.nranks * 2 + _ctx.nranks + dst_rank];
  }
};

}  // namespace

/**
 * @brief Shuffle (all-to-all exchange) packed cudf partitioned table.
 *
 *
 * @param ctx The context of the calling task
 * @param tbl_partitioned The local table partitioned into multiple tables such
 * that `tbl_partitioned.at(i)` should end up at rank i.
 * @param owning_table Optional table owning the data in `tbl_partitioned`.
 * This table is cleaned up early to reduce the peak memory usage.
 * If passed, `tbl_partitioned` is also cleared (as the content is invalid).
 * @return An std::pair where the first entry contains a vector of table_view
 * with all the chunks (including the local copy). The second entry contains
 * a unique_ptr whose contents owns all parts.
 */
std::pair<std::vector<cudf::table_view>,
          std::unique_ptr<std::pair<std::map<int, rmm::device_buffer>, cudf::table>>>
shuffle(TaskContext& ctx,
        std::vector<cudf::table_view>& tbl_partitioned,
        std::unique_ptr<cudf::table> owning_table)
{
  auto context = ctx.get_legate_context();
  if (tbl_partitioned.size() != ctx.nranks) {
    throw std::runtime_error("internal error: partition split has wrong size.");
  }

  // First we pack the columns into contiguous chunks for transfer/shuffling
  // `columns.at(i)` will be send to the i'th task.
  // N.B. all tasks beside itself have a map so `columns.size() == ctx.nranks - 1`.
  std::map<int, cudf::packed_columns> columns;
  for (int i = 0; static_cast<size_t>(i) < tbl_partitioned.size(); ++i) {
    if (i != ctx.rank) {
      columns[i] = cudf::detail::pack(tbl_partitioned[i], ctx.stream(), ctx.mr());
    }
  }
  // Also copy tbl_partitioned.at(ctx.rank).  This copy is unnecessary but allows
  // clearing the (possibly) much larger owning_table (if passed).
  cudf::table local_table(tbl_partitioned.at(ctx.rank), ctx.stream(), ctx.mr());
  if (owning_table) {
    tbl_partitioned.clear();
    owning_table.reset();
  }

  assert(columns.size() == ctx.nranks - 1);
  ExchangedSizes sizes(ctx, columns);

  // Since we a using NCCL, we need to move the metadata of the packed columns to
  // device memory (NCCL only supports GPU and pinned host memory).
  std::map<int, rmm::device_buffer> packed_metadata;
  for (const auto& [peer, col] : columns) {
    packed_metadata.insert(
      {peer,
       rmm::device_buffer(col.metadata->data(), col.metadata->size(), sizes.stream, ctx.mr())});
  }

  // Let's allocate receive buffers for the packed columns.
  // Receive metadata into pinned host memory.
  // Notice, the lifespan of `legate::Buffer` are until the legate task finish,
  // which is fine since we expect the size of the metadata to be small.
  std::map<int, legate::Buffer<uint8_t>> recv_metadata;
  for (int peer = 0; peer < ctx.nranks; ++peer) {
    std::size_t nbytes = sizes.metadata(peer, ctx.rank);
    if (nbytes > 0) {
      assert(peer != ctx.rank);
      // Note: Size of this buffer is taken into account in the mapper:
      recv_metadata.insert(
        {peer, legate::create_buffer<uint8_t>(nbytes, Memory::Kind::Z_COPY_MEM)});
    }
  }
  // Receive gpu_data into device memory (on main task stream).
  std::map<int, rmm::device_buffer> recv_gpu_data;
  for (int peer = 0; peer < ctx.nranks; ++peer) {
    std::size_t nbytes = sizes.gpu_data(peer, ctx.rank);
    if (nbytes > 0) {
      assert(peer != ctx.rank);
      recv_gpu_data.insert({peer, rmm::device_buffer(nbytes, ctx.stream(), ctx.mr())});
    }
  }

  // Perform all-to-all exchange.
  auto task = ctx.get_legate_context();
  task.concurrent_task_barrier();
  CHECK_NCCL(ncclGroupStart());

  // Exchange metadata using the temporary stream `sizes.stream`.
  for (auto& [peer, buf] : recv_metadata) {
    std::size_t nbytes = sizes.metadata(peer, ctx.rank);
    assert(nbytes > 0);
    CHECK_NCCL(ncclRecv(buf.ptr(0), nbytes, ncclInt8, peer, task_nccl(ctx), sizes.stream));
  }
  for (const auto& [peer, buf] : packed_metadata) {
    assert(buf.size() > 0);
    assert(buf.size() == sizes.metadata(ctx.rank, peer));
    CHECK_NCCL(ncclSend(buf.data(), buf.size(), ncclInt8, peer, task_nccl(ctx), sizes.stream));
  }

  // Exchange gpu_data using the task stream `ctx.stream`.
  for (auto& [peer, buf] : recv_gpu_data) {
    std::size_t nbytes = sizes.gpu_data(peer, ctx.rank);
    assert(nbytes > 0);
    CHECK_NCCL(ncclRecv(buf.data(), nbytes, ncclInt8, peer, task_nccl(ctx), ctx.stream()));
  }
  for (const auto& [peer, col] : columns) {
    if (col.gpu_data->size() == 0) { continue; }
    assert(col.gpu_data->size() == sizes.gpu_data(ctx.rank, peer));
    CHECK_NCCL(ncclSend(
      col.gpu_data->data(), col.gpu_data->size(), ncclInt8, peer, task_nccl(ctx), ctx.stream()));
  }
  CHECK_NCCL(ncclGroupEnd());
  task.concurrent_task_barrier();

  // We sync the temporary stream `sizes.stream`, since the unpacking needs the host-side metadata.
  LEGATE_CHECK_CUDA(cudaStreamSynchronize(sizes.stream));

  // Let's unpack and return the packed_columns received from our peers
  // (and our own chunk so that `ret` is ordered for stable sorts)
  std::vector<cudf::table_view> ret;
  for (int peer = 0; peer < ctx.nranks; ++peer) {
    if (peer == ctx.rank) {
      ret.push_back(local_table.view());
      continue;
    }
    uint8_t* gpu_data = nullptr;
    if (recv_gpu_data.count(peer)) {
      gpu_data = static_cast<uint8_t*>(recv_gpu_data.at(peer).data());
    }
    ret.push_back(cudf::unpack(recv_metadata.at(peer).ptr(0), gpu_data));
  }

  using owner_t = std::pair<std::map<int, rmm::device_buffer>, cudf::table>;
  return std::make_pair(
    ret,
    std::make_unique<owner_t>(std::make_pair(std::move(recv_gpu_data), std::move(local_table))));
}

std::unique_ptr<cudf::table> repartition_by_hash(
  TaskContext& ctx,
  const cudf::table_view& table,
  const std::vector<cudf::size_type>& columns_to_hash)
{
  /* The goal is to repartition the table based on the hashing of `columns_to_hash`.
   * Our approach:
   *  1) Each task split their local cudf table into `ctx.nranks` partitions based on the
   *     hashing of `columns_to_hash` and assign each partition to a task.
   *  2) Each task pack (serialize) the partitions not assigned to itself.
   *  4) All tasks shuffle (all-to-all exchange) the partitions. `shuffle` does this by first
   *     packing each partition into a contiguous memory block for the transfer.
   *  5) Each task unpack (deserialize) and concatenate the received columns with the self-assigned
   *     partition.
   *  6) Finally, each task return a new local cudf table that contains the concatenated partitions.
   */

  if (ctx.nranks == 1) {
    // TODO: avoid copy
    return std::make_unique<cudf::table>(table, ctx.stream(), ctx.mr());
  }

  // When used, we need to hold on the partition table as long as tbl_partitioned
  std::unique_ptr<cudf::table> partition_table;
  std::vector<cudf::table_view> tbl_partitioned;
  if (table.num_rows() == 0) {
    tbl_partitioned.reserve(ctx.nranks);
    // cudf seems to have issues with splitting (and maybe hash partitioning) empty tables
    for (int i = 0; i < ctx.nranks; i++) {
      tbl_partitioned.push_back(table);
    }
  } else {
    auto res = cudf::hash_partition(table,
                                    columns_to_hash,
                                    ctx.nranks,
                                    cudf::hash_id::HASH_MURMUR3,
                                    cudf::DEFAULT_HASH_SEED,
                                    ctx.stream(),
                                    ctx.mr());
    partition_table.swap(res.first);

    // Notice, the offset argument for split() and hash_partition() doesn't align. hash_partition()
    // returns the start offset of each partition thus we have to skip the first offset.
    // See: <https://github.com/rapidsai/cudf/issues/4607>.
    auto partition_offsets = std::vector<int>(res.second.begin() + 1, res.second.end());

    tbl_partitioned = cudf::split(*partition_table, partition_offsets, ctx.stream());
  }

  auto [tables, owners] = shuffle(ctx, tbl_partitioned, std::move(partition_table));

  return cudf::concatenate(tables, ctx.stream(), ctx.mr());
}

}  // namespace legate::dataframe::task
