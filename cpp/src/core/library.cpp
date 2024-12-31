/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <vector>

#include <legate_dataframe/core/allocator.hpp>
#include <legate_dataframe/core/library.hpp>

namespace legate::dataframe {
namespace task {

legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}
}  // namespace task

namespace {

constexpr auto library_name = "legate_dataframe";

Legion::Logger logger(library_name);

class Mapper : public legate::mapping::Mapper {
 public:
  Mapper() {}

  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    using legate::mapping::StoreMapping;
    std::vector<StoreMapping> mappings;

    // For now, we set "exact" policy for all Stores
    // TODO: only use "exact" when needed
    for (const legate::mapping::Array& ary : task.inputs()) {
      if (ary.type().variable_size()) { continue; }
      for (const legate::mapping::Store& store : ary.stores()) {
        mappings.push_back(
          StoreMapping::default_mapping(store, options.front(), /*exact = */ true));
      }
    }
    for (const legate::mapping::Array& ary : task.outputs()) {
      if (ary.type().variable_size()) { continue; }
      for (const legate::mapping::Store& store : ary.stores()) {
        mappings.push_back(
          StoreMapping::default_mapping(store, options.front(), /*exact = */ true));
      }
    }
    return mappings;
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override { return legate::Scalar{0}; }

 private:
  const legate::mapping::MachineQueryInterface* machine_;
};

legate::Library create_and_registrate_library()
{
  const char* env = std::getenv("LDF_DISABLE_GLOBAL_MEMORY_RESOURCE");
  if (env == nullptr || std::string{env} == "0") {
    GlobalMemoryResource::set_as_default_mmr_resource();
  }
  auto context = legate::Runtime::get_runtime()->find_or_create_library(
    library_name, legate::ResourceConfig{}, std::make_unique<Mapper>());
  task::Registry::get_registrar().register_all_tasks(context);
  return legate::Runtime::get_runtime()->find_library(legate::dataframe::library_name);
}

}  // namespace

legate::Library& get_library()
{
  static legate::Library library = create_and_registrate_library();
  return library;
}

}  // namespace legate::dataframe
