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

#include <vector>

#include <legate.h>

#include <legate_dataframe/core/task_context.hpp>

namespace legate::dataframe::argument {

/**
 * @brief Add scalar to the next task argument
 *
 * This should match a call to `get_next_scalar()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @tparam T The type of the scalar value.
 * @param task The legate task to add the argument.
 * @param scalar The scalar.
 */
template <typename T>
void add_next_scalar(legate::AutoTask& task, const T& scalar)
{
  task.add_scalar_arg(legate::Scalar(scalar));
}

template <>
inline void add_next_scalar<legate::Scalar>(legate::AutoTask& task, const legate::Scalar& scalar)
{
  task.add_scalar_arg(scalar);
}

/**
 * @brief Add a vector of scalars to the next task argument
 *
 * This should match a call to `get_next_scalar_vector()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @tparam T The type of the scalar values.
 * @param task The legate task to add the argument.
 * @param items The vector of scalars.
 */
template <typename T>
void add_next_scalar_vector(AutoTask& task, const std::vector<T>& scalars)
{
  add_next_scalar(task, legate::Scalar(scalars));
}

template <>
inline void add_next_scalar_vector(AutoTask& task, const std::vector<legate::Rect<1>>& scalars)
{
  /* Ironically, legate 25.08 has issues with legate scalars. */
  auto arr_type = legate::fixed_array_type(legate::rect_type(1), scalars.size());
  add_next_scalar(task, legate::Scalar(arr_type, scalars.data(), /* copy */ true));
}

template <>
inline void add_next_scalar_vector(AutoTask& task, const std::vector<std::string>& scalars)
{
  // String vectors don't work as scalars directly in legate as of 2025-08.
  std::stringstream ss;
  std::vector<size_t> lengths;
  lengths.reserve(scalars.size());
  for (auto& string : scalars) {
    ss << string;
    lengths.emplace_back(string.length());
  }
  add_next_scalar_vector(task, lengths);
  add_next_scalar(task, ss.str());
}

/**
 * @brief Get next scalar task argument
 *
 * This should match a call to `add_next_scalar()` by a legate client.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @tparam T The type of the scalar value.
 * @param ctx The task context active in the calling task.
 * @return The value of the scalar argument.
 */
template <typename T>
T get_next_scalar(TaskContext& ctx)
{
  return ctx.get_next_scalar_arg().template value<T>();
}

/**
 * @brief Get next vector of scalars argument
 *
 * This should match a call to `add_next_scalar_vector()` by a legate client.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * Note: In the future we may change this to not be a vector, but quite a few
 * places expect a vector (of non const values).
 *
 * @tparam T The type of the scalar values.
 * @param ctx The task context active in the calling task.
 * @return The vector of scalar values.
 */
template <typename T>
std::vector<T> get_next_scalar_vector(TaskContext& ctx)
{
  auto items = ctx.get_next_scalar_arg().values<T>();
  std::vector<T> ret;
  ret.reserve(items.size());
  for (auto& item : items) {
    ret.emplace_back(item);
  }
  return ret;
}

template <>
inline std::vector<std::string> get_next_scalar_vector(TaskContext& ctx)
{
  std::vector<std::string> ret;
  auto lengths = get_next_scalar_vector<std::size_t>(ctx);
  auto strings = ctx.get_next_scalar_arg().value<std::string>();

  ret.reserve(lengths.size());
  size_t start = 0;
  for (auto& len : lengths) {
    std::string substr(strings, start, len);
    ret.emplace_back(substr);
    start += len;
  }
  return ret;
}

/**
 * @brief Get next input task argument
 *
 * The default non-specialized implementation has been delete.
 * Instead each argument type (such as PhysicalColumn or PhysicalTable)
 * implements their own specialization.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @tparam T The type of the input task argument.
 * @param ctx The task context active in the calling task.
 * @return The input task argument.
 */
template <typename T>
T get_next_input(TaskContext& ctx) = delete;

/**
 * @brief Get next output task argument
 *
 * The default non-specialized implementation has been delete.
 * Instead each argument type (such as PhysicalColumn or PhysicalTable)
 * implements their own specialization.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @tparam T The type of the output task argument.
 * @param ctx The task context active in the calling task.
 * @return The output task argument.
 */
template <typename T>
T get_next_output(TaskContext& ctx) = delete;

/**
 * @brief Adding alignment constraints to a task
 *
 * Use this to make sure that task arguments such as LogicalArray, LogicalColumn,
 * and LogicalTable uses the same partitioning (i.e they are distributed between legate
 * nodes in the same way).
 *
 * @param task The task in which the alignment constraints apply.
 * @param variables The variables to align. Typically, returned by "add_next_*" functions.
 */
void add_alignment_constraints(legate::AutoTask& task,
                               const std::vector<legate::Variable>& variables);

/**
 * @brief Force legate to launch `min_num_tasks`
 *
 * With <https://github.com/nv-legate/legate.core.internal/pull/376>, legate will only launch
 * parallel tasks when the task input is greater than `LEGATE_MIN_GPU_CHUNK`. This is a problem when
 * launching tasks that doesn't take any array input, such as the CSV reader. In such cases, we add
 * an untouched "dummy" column as input.
 *
 * NB: like the other argument functions, this function should match a `get_parallel_launch_task()`
 *     call in the task.
 *
 * @param task The task in which the alignment constraints apply.
 */
void add_parallel_launch_task(legate::AutoTask& task, int min_num_tasks);

/**
 * @brief Force legate to launch one task per GPU available
 *
 * With <https://github.com/nv-legate/legate.core.internal/pull/376>, legate will only launch
 * parallel tasks when the task input is greater than `LEGATE_MIN_GPU_CHUNK`. This is a problem when
 * launching tasks that doesn't take any array input, such as otheur CSV reader. In such cases, we
 * add an untouched "dummy" column as input.
 *
 * NB: like the other argument functions, this function should match a `get_parallel_launch_task()`
 *     call in the task.
 */
void add_parallel_launch_task(legate::AutoTask& task);

/*
 * Similar to `add_parallel_launch_task()`, but an `Nx1` dimensional array because
 * different dimensions don't seem to mix well right now (as of legate 25.05).
 */
void add_parallel_launch_task_2d(legate::AutoTask& task);

/**
 * @brief Handle of task launched using `add_parallel_launch_task()`
 */
void get_parallel_launch_task(TaskContext& ctx);

}  // namespace legate::dataframe::argument
