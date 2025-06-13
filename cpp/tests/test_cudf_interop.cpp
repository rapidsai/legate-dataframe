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

#include <legate.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>

using namespace legate::dataframe;

namespace {

static const char* library_name = "test.cudf_interop";

struct RoundTripTableTask : public legate::LegateTask<RoundTripTableTask> {
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
  static inline const auto TASK_CONFIG      = legate::TaskConfig{legate::LocalTaskID{0}};

  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};

    const auto input = argument::get_next_input<task::PhysicalTable>(ctx);
    auto output      = argument::get_next_output<task::PhysicalTable>(ctx);
    auto copy        = std::make_unique<cudf::table>(input.table_view(), ctx.stream(), ctx.mr());
    output.move_into(std::move(copy));
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  RoundTripTableTask::register_variants(context);
}

void round_trip(const LogicalTable& input, const LogicalTable& output)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, RoundTripTableTask::TASK_CONFIG.task_id());

  // Launch task
  argument::add_next_input(task, input);
  argument::add_next_output(task, output);
  runtime->submit(std::move(task));
}
}  // namespace

template <typename T>
struct CudfInterOp : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(CudfInterOp, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(CudfInterOp, RoundTrip)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({0, 1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<TypeParam> b({5, 6, 7, 8, 9}, {1, 0, 1, 1, 1});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table original(std::move(cols));

  LogicalTable input(original, {"a", "b"});
  LogicalTable output = LogicalTable::empty_like(input);
  assert(input.get_column(0).cudf_type() == cudf::data_type{cudf::type_to_id<TypeParam>()});
  assert(output.get_column(0).cudf_type() == cudf::data_type{cudf::type_to_id<TypeParam>()});

  round_trip(input, output);

  CUDF_TEST_EXPECT_TABLES_EQUAL(original, input.get_cudf()->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(original, output.get_cudf()->view());
}

TYPED_TEST(CudfInterOp, RoundTripEmpty)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::fixed_width_column_wrapper<TypeParam> a({});
  auto empty_mask = rmm::device_uvector<bool>(0, cudf::get_default_stream());
  cudf::test::fixed_width_column_wrapper<TypeParam> b({}, empty_mask.begin());
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table original(std::move(cols));

  LogicalTable input(original, {"a", "b"});
  LogicalTable output = LogicalTable::empty_like(input);
  round_trip(input, output);

  CUDF_TEST_EXPECT_TABLES_EQUAL(original, input.get_cudf()->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(original, output.get_cudf()->view());
}

TEST(CudfInterOpStrings, RoundTrip)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cudf::test::strings_column_wrapper a({"", "this", "", "is", "a", "test"});
  cudf::test::strings_column_wrapper b({"hi", "this", "is", "", "a", "string"}, {0, 1, 1, 0, 0, 1});
  cols.push_back(a.release());
  cols.push_back(b.release());
  cudf::table original(std::move(cols));

  LogicalTable input(original, {"a", "b"});
  LogicalTable output = LogicalTable::empty_like(input);
  round_trip(input, output);

  CUDF_TEST_EXPECT_TABLES_EQUAL(original, input.get_cudf()->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(original, output.get_cudf()->view());
}

TEST(CudfInterOpStrings, EmptyRoundTrip)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_empty_column(cudf::type_id::STRING));
  cudf::table original(std::move(cols));

  LogicalTable input(original, {"a"});
  LogicalTable output = LogicalTable::empty_like(input);
  round_trip(input, output);

  CUDF_TEST_EXPECT_TABLES_EQUAL(input.get_cudf()->view(), output.get_cudf()->view());
}

TYPED_TEST(CudfInterOp, ScalarConversion)
{
  using ScalarType = cudf::scalar_type_t<TypeParam>;

  TypeParam scalar_val = cudf::test::make_type_param_scalar<TypeParam>(5);
  auto scalar          = ScalarType(scalar_val, true);
  auto scalar_null     = ScalarType(scalar_val, false);
  auto converted       = legate::dataframe::LogicalColumn(scalar);
  auto converted_null  = legate::dataframe::LogicalColumn(scalar_null);

  EXPECT_TRUE(converted.is_scalar());
  EXPECT_TRUE(converted_null.is_scalar());

  auto result =
    converted.get_cudf_scalar(cudf::get_default_stream(), rmm::mr::get_current_device_resource());
  auto result_null = converted_null.get_cudf_scalar(cudf::get_default_stream(),
                                                    rmm::mr::get_current_device_resource());

  EXPECT_EQ(result->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});
  EXPECT_EQ(result_null->type(), cudf::data_type{cudf::type_to_id<TypeParam>()});

  EXPECT_TRUE(result->is_valid());
  EXPECT_FALSE(result_null->is_valid());

  auto typed_result = static_cast<ScalarType*>(result.get());
  EXPECT_EQ(typed_result->value(), scalar_val);
}
