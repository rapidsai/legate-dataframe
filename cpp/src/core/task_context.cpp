/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate {
namespace dataframe {

namespace {
int get_rank(legate::TaskContext& context)
{
  if (context.is_single_task()) { return 0; }

  const auto task_index    = context.get_task_index();
  const auto launch_domain = context.get_launch_domain();
  return linearize(launch_domain.lo(), launch_domain.hi(), task_index);
}
}  // namespace

TaskContext::TaskContext(legate::TaskContext& context,
                         size_t arg_scalar_idx,
                         size_t arg_input_idx,
                         size_t arg_output_idx)
  : _context{context},
    _arg_scalar_idx{arg_scalar_idx},
    _arg_input_idx{arg_input_idx},
    _arg_output_idx{arg_output_idx},
    rank(get_rank(context)),
    nranks(_context.get_launch_domain().get_volume())
{
}

}  // namespace dataframe
}  // namespace legate
