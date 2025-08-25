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

#include <legate.h>

#include <nccl.h>

#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::task {

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
    exit(error);
  }
}

/**
 * @brief Get the NCCL task handle
 *
 * This assumes that the calling task was created with a NCCL communicator,
 * see `legate::AutoTask::add_communicator()`.
 *
 * Notice, if a task launch ends up emitting only a single point task, that task will
 * not get passed a communicator, even if one was requested at task launching time.
 * In this case, a `std::out_of_range` exception is thrown.
 *
 * @param ctx The context of the calling task
 * @param idx The index of the NCCL communicator i.e. the order it was added to the task.
 * @return The NCCL communicator
 * @throw  std::out_of_range if @p idx is an invalid index.
 */
ncclComm_t task_nccl(const TaskContext& ctx, size_t idx = 0)
{
  return *ctx.get_legate_context().communicators().at(idx).get<ncclComm_t*>();
}

}  // namespace legate::dataframe::task
