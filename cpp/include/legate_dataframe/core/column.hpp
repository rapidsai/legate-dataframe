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

#include <optional>
#include <stdexcept>
#include <string>

#include <arrow/api.h>

#ifdef LEGATE_DATAFRAME_USE_CUDA
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#endif

#include <legate.h>

#include <legate_dataframe/core/task_argument.hpp>
#include <legate_dataframe/core/task_context.hpp>
#include <legate_dataframe/utils.hpp>

namespace legate::dataframe {

// Make this function available to cuda code
// Can likely be removed in future
namespace detail {
legate::LogicalArray from_arrow(std::shared_ptr<arrow::Array> arrow_array, bool scalar = false);
};

/**
 * @brief Logical column
 *
 * Underlying a logical column is a logical array. The column doesn't own the array,
 * a logical array can be part of multiple columns.
 *
 * Note that the columns have a `scalar()` attribute.  If true the column is always
 * length 1 and will be treated as a scalar, i.e. with broadcast semantics, in
 * some contexts.
 * We do not allow broadcast semantics more generally, because checking for
 * length 1 may be a blocking operation and we need this information at task launch.
 */
class LogicalColumn {
 public:
  /**
   * @brief Create an undefined column only to used for copying and moving
   *
   * This ctor is only here because of Cython
   */
  LogicalColumn() = default;

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Create a column with a legate array as the data
   *
   * @param array The logical array (zero copy)
   * @param cudf_type The cudf data type of the column. If `EMPTY` (default), the cudf data type is
   * derived from the data type of `array`.
   * @param scalar Whether to consider this column scalar.  WARNING: currently
   * it is the callers responsibility to ensure the array is length 1 as this
   * check could be blocking.
   */
  LogicalColumn(legate::LogicalArray array,
                cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY},
                bool scalar               = false)
    : array_{std::move(array)}, scalar_{scalar}
  {
    if (array_->dim() != 1) { throw std::invalid_argument("array must be 1-D"); }
    // Note: Checking the volume could be blocking, so assume that this is fine.
    assert(!scalar || array_->unbound() || array_->volume() == 1);

    if (cudf_type.id() == cudf::type_id::EMPTY) {
      arrow_type_ = to_arrow_type(array_->type().code());
    } else {
      arrow_type_ = to_arrow_type(cudf_type.id());
    }
  }
#endif

  /**
   * @brief Create a column with a legate array as the data
   *
   * @param array The logical array (zero copy)
   * @param data_type The arrow data type of the column. If nullptr, the arrow data type is
   * derived from the data type of `array`.
   * @param scalar Whether to consider this column scalar.  WARNING: currently
   * it is the callers responsibility to ensure the array is length 1 as this
   * check could be blocking.
   */
  LogicalColumn(legate::LogicalArray array,
                std::shared_ptr<arrow::DataType> data_type,
                bool scalar = false)
    : array_{std::move(array)}, scalar_{scalar}
  {
    if (array_->dim() != 1) { throw std::invalid_argument("array must be 1-D"); }
    // Note: Checking the volume could be blocking, so assume that this is fine.
    assert(!scalar || array_->unbound() || array_->volume() == 1);

    if (!data_type) {
      arrow_type_ = to_arrow_type(array_->type().code());
    } else {
      arrow_type_ = data_type;
    }
  }

  /*
   * Convenience constructor for tests
   */
  template <typename T>
  LogicalColumn(const std::vector<T>& data,
                const std::vector<bool>& null_mask                  = {},
                bool scalar                                         = false,
                typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr)
  {
    const size_t nbytes   = data.size() * sizeof(T);
    auto runtime          = legate::Runtime::get_runtime();
    auto legate_type_code = legate::type_code_of_v<T>;
    auto data_store =
      runtime->create_store({data.size()}, legate::primitive_type(legate_type_code), false);
    auto ptr = data_store.get_physical_store().template write_accessor<T, 1, false>().ptr(0);
    std::copy(data.begin(), data.end(), ptr);

    if (null_mask.empty()) {
      array_ = std::move(legate::LogicalArray(data_store));
    } else {
      auto null_mask_store = runtime->create_store({data.size()}, legate::bool_(), false);
      auto null_mask_ptr =
        null_mask_store.get_physical_store().template write_accessor<bool, 1, false>().ptr(0);
      std::copy(null_mask.begin(), null_mask.end(), null_mask_ptr);
      array_ = std::move(legate::LogicalArray(data_store, null_mask_store));
    }
    scalar_     = scalar;
    arrow_type_ = to_arrow_type(legate_type_code);
  }

  /*
   * Convenience constructor for tests
   */
  LogicalColumn(const std::vector<std::string>& data,
                const std::vector<bool>& null_mask = {},
                bool scalar                        = false)
  {
    auto runtime       = legate::Runtime::get_runtime();
    std::size_t nbytes = 0;
    std::vector<std::size_t> offsets(1);
    for (const auto& str : data) {
      nbytes += str.size();
      offsets.push_back(nbytes);
    }
    // Create store for ranges
    auto ranges_store = runtime->create_store({data.size()}, legate::rect_type(1), false);
    auto data_store   = runtime->create_store({nbytes}, legate::int8(), false);

    // Copy the data
    auto data_ptr =
      data_store.get_physical_store().template write_accessor<int8_t, 1, false>().ptr(0);
    auto ranges_ptr =
      ranges_store.get_physical_store().template write_accessor<legate::Rect<1>, 1, false>().ptr(0);
    for (size_t i = 0; i < data.size(); ++i) {
      std::copy(data[i].data(), data[i].data() + data[i].size(), data_ptr + offsets[i]);
      ranges_ptr[i] = legate::Rect<1>(offsets[i], offsets[i + 1] - 1);
    }

    if (null_mask.empty()) {
      array_ = runtime->create_string_array(legate::LogicalArray(ranges_store),
                                            legate::LogicalArray(data_store));
    } else {
      auto null_mask_store = runtime->create_store({data.size()}, legate::bool_(), false);

      auto null_mask_ptr =
        null_mask_store.get_physical_store().template write_accessor<bool, 1, false>().ptr(0);
      std::copy(null_mask.begin(), null_mask.end(), null_mask_ptr);
      array_ = runtime->create_string_array(legate::LogicalArray(ranges_store, null_mask_store),
                                            legate::LogicalArray(data_store));
    }
    scalar_     = scalar;
    arrow_type_ = arrow::utf8();
  }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Create a column from a local cudf column
   *
   * This call blocks the client's control flow and scatter the data to all
   * legate nodes.
   *
   * @param cudf_col The local cuDF column to copy into a logical column
   * @param stream CUDA stream used for device memory operations
   */
  LogicalColumn(cudf::column_view cudf_col,
                rmm::cuda_stream_view stream = cudf::get_default_stream());
#endif

  /**
   * @brief Create a column from a local arrow array
   *
   * This call blocks the client's control flow and scatter the data to all
   * legate nodes.
   *
   * @param arrow_array The local arrow array to copy into a logical column
   */
  LogicalColumn(std::shared_ptr<arrow::Array> arrow_array);

  /**
   * @brief Create a scalar column from a local arrow scalar
   *
   * This call blocks the client's control flow and scatter the data to all
   * legate nodes.
   * This column will be marked and treated as scalar.
   *
   * @param arrow_scalar The local arrow scalar to copy into a logical column
   */
  LogicalColumn(std::shared_ptr<arrow::Scalar> arrow_scalar);

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Create a scalar column from a local cudf scalar
   *
   * This call blocks the client's control flow and scatter the data to all
   * legate nodes.
   * This column will be marked and treated as scalar.
   *
   * @param cudf_scalar The local cuDF scalar to copy into a logical column
   * @param stream CUDA stream used for device memory operations
   */
  LogicalColumn(const cudf::scalar& cudf_scalar,
                rmm::cuda_stream_view stream = cudf::get_default_stream());
#endif

  /**
   * @brief Create a new column from an existing column
   *
   * This function always returns a non-scalar column, even if the input was
   * considered scalar.  Functions that wish to propagate scalar information
   * must use the long signature.
   *
   * @param other The prototype column
   * @return The new unbounded column with the type and nullable equal `other`
   */
  static LogicalColumn empty_like(const LogicalColumn& other)
  {
    return LogicalColumn(legate::Runtime::get_runtime()->create_array(
                           other.array_->type(), other.array_->dim(), other.array_->nullable()),
                         other.arrow_type(),
                         false);
  }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Create a new unbounded column from an existing local cuDF column
   *
   * @param other The prototype column
   * @return The new unbounded column with the type and nullable equal `other`
   */
  static LogicalColumn empty_like(const cudf::column_view& other)
  {
    return LogicalColumn(legate::Runtime::get_runtime()->create_array(
                           to_legate_type(other.type().id()), 1, other.nullable()),
                         other.type());
  }
#endif

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Create a new unbounded column from dtype and nullable
   *
   * @param dtype The data type of the new column
   * @param nullable The nullable of the new column
   * @param scalar Whether the result is a scalar column.
   * @return The new unbounded column
   */
  static LogicalColumn empty_like(cudf::data_type dtype,
                                  bool nullable,
                                  bool scalar                = false,
                                  std::optional<size_t> size = std::nullopt)
  {
    if (!size.has_value()) {
      return LogicalColumn(
        legate::Runtime::get_runtime()->create_array(to_legate_type(dtype.id()), 1, nullable),
        dtype,
        scalar);
    } else {
      return LogicalColumn(legate::Runtime::get_runtime()->create_array(
                             Shape{size.value()}, to_legate_type(dtype.id()), nullable),
                           dtype,
                           scalar);
    }
  }
#endif

  /**
   * @brief Create a new unbounded column from dtype and nullable
   *
   * @param dtype The data type of the new column
   * @param nullable The nullable of the new column
   * @param scalar Whether the result is a scalar column.
   * @return The new unbounded column
   */
  static LogicalColumn empty_like(std::shared_ptr<arrow::DataType> dtype,
                                  bool nullable,
                                  bool scalar                = false,
                                  std::optional<size_t> size = std::nullopt)
  {
    if (!size.has_value()) {
      return LogicalColumn(
        legate::Runtime::get_runtime()->create_array(to_legate_type(*dtype), 1, nullable),
        dtype,
        scalar);
    } else {
      return LogicalColumn(legate::Runtime::get_runtime()->create_array(
                             Shape{size.value()}, to_legate_type(*dtype), nullable),
                           dtype,
                           scalar);
    }
  }

 public:
  LogicalColumn(const LogicalColumn& other)            = default;
  LogicalColumn& operator=(const LogicalColumn& other) = default;
  LogicalColumn(LogicalColumn&& other)                 = default;
  LogicalColumn& operator=(LogicalColumn&& other)      = default;

 public:
  /**
   * @brief Return the underlying logical array
   *
   * @return The underlying logical array
   */
  legate::LogicalArray get_logical_array() const { return *array_; }

  /**
   * @brief Creates a physical array for the underlying logical array
   *
   * This call blocks the client's control flow and fetches the data for the whole
   * array to the current node.
   *
   * @return A physical array of the underlying logical array
   */
  legate::PhysicalArray get_physical_array() const { return array_->get_physical_array(); }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Copy the logical column into a local cudf column
   *
   * This call blocks the client's control flow and fetches the data for the
   * whole column to the current node.
   *
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations.
   * @return cudf column, which own the data
   */
  std::unique_ptr<cudf::column> get_cudf(
    rmm::cuda_stream_view stream        = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;
#endif

  /**
   * @brief Copy the logical column into a local arrow array
   *
   * This call blocks the client's control flow and fetches the data for the
   * whole column to the current node.
   *
   * @return arrow array, which own the data
   */
  std::shared_ptr<arrow::Array> get_arrow() const;

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Copy the logical column into a local cudf scalar
   *
   * This call blocks the client's control flow and fetches the data for the
   * whole column to the current node.
   *
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations.
   * @throws std::invalid_argument if this is not a length 1/scalar column.
   * @return cudf scalar, which own the data
   */
  std::unique_ptr<cudf::scalar> get_cudf_scalar(
    rmm::cuda_stream_view stream        = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) const;

#endif

  /**
   * @brief Offload column to the specified target memory (default SYSMEM).
   *
   * This method offloads the underlying data to the specified target memory.
   * The purpose of this is to free up GPU memory resources.
   * See `legate::LogicalArray::offload_to` for more information.
   *
   * @param target_mem The `legate::mapping::StoreTarget` target memory.
   * This will be `legate::mapping::StoreTarget::SYSMEM` to move data to the CPU.
   */
  void offload_to(legate::mapping::StoreTarget target_mem) const
  {
    return array_->offload_to(target_mem);
  }

  /**
   * @brief Indicates whether the column is unbound
   *
   * @return true The column is unbound
   * @return false The column is bound
   */
  [[nodiscard]] bool unbound() const { return array_->unbound(); }

  /**
   * @brief Get the data type of the underlying logical array
   *
   * @return The legate data type
   */
  [[nodiscard]] legate::Type type() const { return array_->type(); }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Get the cudf data type of the column
   *
   * @return The cudf data type
   */
  [[nodiscard]] cudf::data_type cudf_type() const { return to_cudf_type(arrow_type_); }
#endif

  /**
   * @brief Get the arrow data type of the column
   *
   * @return The arrow data type
   */
  [[nodiscard]] std::shared_ptr<arrow::DataType> arrow_type() const { return arrow_type_; }

  /**
   * @brief Indicates whether the array is nullable
   *
   * @return true The array is nullable
   * @return false The array is non-nullable
   */
  [[nodiscard]] bool nullable() const { return array_->nullable(); }

  /**
   * @brief Returns the number of rows
   *
   * @throw std::runtime_error if column is unbound. std::logic_error if a scalar
   * but the number of rows isn't one.
   * @return The number of rows
   */
  [[nodiscard]] std::size_t num_rows() const
  {
    if (unbound()) {
      throw std::runtime_error("Cannot call `.num_rows()` on a unbound LogicalColumn");
    }
    auto rows = array_->volume();
    if (is_scalar() && rows != 1) {
      throw std::logic_error("PhysicalColumn is scalar but doesn't have one row.");
    }
    return rows;
  }

  /**
   * @brief Return true if the column is considered scalar.
   *
   * A scalar column always has a size of one.  Scalar columns are mostly supported
   * for binary operations and have scalar/broadcast semantics if supported.
   * (For unsupported operations they may work, but behave like a length 1 column).
   *
   * @return true if scalar, otherwise false.
   */
  bool is_scalar() const { return scalar_; };

  /**
   * @brief Return a printable representational string
   *
   * @param max_num_items Maximum number of items to include before items are abbreviated.
   * @return Printable representational string
   */
  std::string repr(size_t max_num_items = 30) const;

  /**
   * @brief Slice the column
   *
   * @param slice The Legate slice into the column.  Supports negative values,
   * `Slice::OPEN`, and does not include stop index.
   * @return The sliced column
   */
  [[nodiscard]] LogicalColumn slice(const legate::Slice& slice) const;

 private:
  // In order to support a default ctor and assignment (used by Cython),
  // we make the legate array optional and the rest non-const.
  std::optional<legate::LogicalArray> array_;
  std::shared_ptr<arrow::DataType> arrow_type_;
  bool scalar_{false};
};

namespace task {

/**
 * @brief Local physical column used in tasks
 */
class PhysicalColumn {
 public:
  /**
   * @brief Create a column with a legate array as the data
   *
   * @param ctx The context of the calling task
   * @param array The logical array (zero copy)
   * @param cudf_type The cudf data type of the column
   * @param scalar_out Indicate that the result must be scalar (so cannot bind
   * more data).
   * column is part of. Use a negative value to indicate that the number of rows is
   * unknown.
   */
  PhysicalColumn(TaskContext& ctx,
                 legate::PhysicalArray array,
                 std::shared_ptr<arrow::DataType> arrow_type,
                 bool scalar_out = false)
    : ctx_{&ctx}, array_{std::move(array)}, arrow_type_(arrow_type), scalar_out_{scalar_out}
  {
  }

 public:
  PhysicalColumn(const PhysicalColumn& other)            = delete;
  PhysicalColumn& operator=(const PhysicalColumn& other) = delete;
  PhysicalColumn(PhysicalColumn&& other)                 = default;
  PhysicalColumn& operator=(PhysicalColumn&& other)      = delete;

 public:
  /**
   * @brief Indicates whether the column is unbound or not
   *
   * For string columns the underlying characters will be (or can be) unbound
   * even if the ranges are bound.
   *
   * @return true The column is unbound
   * @return false The column is bound
   */
  [[nodiscard]] bool unbound() const
  {
    if (!array_.nested()) { return array_.data().is_unbound_store(); }
    // New string columns won't really have bound characters so only check ranges.
    // (only nested dtypes we have right now)
    return array_.as_string_array().ranges().data().is_unbound_store();
  }

  /**
   * @brief Get the data type of the underlying logical array
   *
   * @return The legate data type
   */
  [[nodiscard]] legate::Type type() const { return array_.type(); }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Get the cudf data type of the column
   *
   * @return The cudf data type
   */
  [[nodiscard]] cudf::data_type cudf_type() const { return to_cudf_type(arrow_type_); }
#endif

  [[nodiscard]] std::shared_ptr<arrow::DataType> arrow_type() const { return arrow_type_; }

  /**
   * @brief Indicates whether the column is nullable
   *
   * @return true The column is nullable
   * @return false The column is non-nullable
   */
  [[nodiscard]] bool nullable() const { return array_.nullable(); }

  /**
   * @brief Returns the number of rows
   *
   * @throw std::runtime_error if column is unbound. std::logic_error if a scalar
   * but the number of rows isn't one.
   * @return The number of rows
   */
  [[nodiscard]] std::size_t num_rows() const
  {
    if (unbound()) {
      throw std::runtime_error(
        "Cannot call `.num_rows()` on a unbound PhysicalColumn, please bind it using "
        "`.move_into()`");
    }
    auto rows = array_.shape<1>().volume();
    if (scalar_out_ && rows != 1) {
      throw std::logic_error("PhysicalColumn is scalar but doesn't have one row.");
    }
    return rows;
  }

  /**
   * @brief Returns the row offset relative to the logical column this physical column is part of.
   *
   * The physical column `x` represent the following rows of the logical column given as task input:
   *   `x.global_row_offset()` .. `x.global_row_offset() + x.num_rows()`.
   *
   * @throw std::runtime_error if column is unbound.
   * @return The row offset in number of rows (inclusive)
   */
  [[nodiscard]] int64_t global_row_offset() const
  {
    if (unbound()) {
      throw std::runtime_error(
        "Cannot call `.global_row_offset()` on a unbound PhysicalColumn, please bind it using "
        "`.move_into()`");
    }
    return array_.shape<1>().lo[0];
  }

  /**
   * @brief Returns true if the data is partitioned.
   *
   * You can use this to check whether a column is partitioned, please see
   * `legate::PhysicalStore::is_partitioned` for more information.
   * This can be used to check whether a column is broadcasted (i.e. partitioned
   * is false), meaning that all workers see the same data.
   *
   * @return true if data is partitioned.
   */
  [[nodiscard]] bool is_partitioned() const
  {
    if (!array_.nested()) { return array_.data().is_partitioned(); }
    return array_.as_string_array().ranges().data().is_partitioned();
  }

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Return a cudf column view of this physical column
   *
   * NB: The physical column MUST outlive the returned view thus it is UB to do some-
   *     thing like `argument::get_next_input<PhysicalColumn>(ctx).column_view();`
   *
   * @throw cudf::logic_error if column is unbound.
   * @return A new column view.
   */
  cudf::column_view column_view() const;
#endif

  std::shared_ptr<arrow::Array> arrow_array_view() const;

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Return a cudf scalar for physical column
   *
   * NB: The physical column MUST outlive the returned scalar thus it is UB to do some-
   *     thing like `argument::get_next_input<PhysicalColumn>(ctx).cudf_scalar();`
   *
   * Note that the above should be considered true even if currently the scalar
   * may not view the memory (you must consider the scalar immutable).
   *
   * @throw cudf::logic_error if column is unbound or the size is not one.
   * @return A new cudf scalar.
   */
  std::unique_ptr<cudf::scalar> cudf_scalar() const;
#endif

  /**
   * @brief Return a printable representational string
   *
   * @param max_num_items Maximum number of items to include before items are abbreviated.
   * @return Printable representational string
   */
  std::string repr(legate::Memory::Kind mem_kind,
                   cudaStream_t stream,
                   size_t max_num_items = 30) const;

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Copy local cudf column into this unbound physical column
   *
   * @param column The cudf column to copy
   */
  void copy_into(std::unique_ptr<cudf::column> column);

  /**
   * @brief Copy local cudf scalar into this bound physical column
   *
   * @param scalar The cudf scalar to copy
   */
  void copy_into(std::unique_ptr<cudf::scalar> scalar);
#endif

  /**
   * @brief Copy local arrow array into this unbound physical column
   *
   * @param column The arrow array to copy
   */
  void copy_into(std::shared_ptr<arrow::Array> column);

#ifdef LEGATE_DATAFRAME_USE_CUDA
  /**
   * @brief Move local cudf column into this unbound physical column
   *
   * @param column The cudf column to move
   */
  void move_into(std::unique_ptr<cudf::column> column);

  /**
   * @brief Move local cudf scalar into this unbound physical column
   *
   * @param scalar The cudf scalar to move
   */
  void move_into(std::unique_ptr<cudf::scalar> scalar);
#endif

  /**
   * @brief Move arrow array into this unbound physical column
   *
   * @param column The arrow array to move
   */
  void move_into(std::shared_ptr<arrow::Array> column);

  /**
   * @brief Makes the unbound column empty. Valid only when the column is unbound.
   */
  void bind_empty_data() const;

 private:
  TaskContext* ctx_;
  legate::PhysicalArray array_;
  const std::shared_ptr<arrow::DataType> arrow_type_;
#ifdef LEGATE_DATAFRAME_USE_CUDA
  mutable std::vector<std::unique_ptr<cudf::column>> tmp_cols_;
  mutable std::vector<rmm::device_buffer> tmp_null_masks_;
#endif
  const bool scalar_out_;  // scalar output checks binding has size 1.
};
}  // namespace task

namespace argument {

/**
 * @brief Add a logical column to the next input task argument
 *
 * This should match a call to `get_next_input<PhysicalColumn>()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @param task The legate task to add the argument.
 * @param tbl The logical column to add as the next task argument.
 * @param broadcast If set to true, each worker is guaranteed to get a copy
 * of the data.
 */
legate::Variable add_next_input(legate::AutoTask& task,
                                const LogicalColumn& col,
                                bool broadcast = false);

/**
 * @brief Add a logical column to the next output task argument
 *
 * This should match a call to `get_next_input<PhysicalColumn>()` by a legate task.
 *
 * NB: the order of "add_next_*" calls must match the order of the
 * corresponding "get_next_*" calls.
 *
 * @param task The legate task to add the argument.
 * @param tbl The logical column to add as the next task argument.
 */
legate::Variable add_next_output(legate::AutoTask& task, const LogicalColumn& col);

template <>
inline task::PhysicalColumn get_next_input<task::PhysicalColumn>(TaskContext& ctx)
{
  auto arrow_type = argument::get_next_scalar<std::shared_ptr<arrow::DataType>>(ctx);
  return task::PhysicalColumn(ctx, ctx.get_next_input_arg(), arrow_type);
}

template <>
inline task::PhysicalColumn get_next_output<task::PhysicalColumn>(TaskContext& ctx)
{
  auto arrow_type = argument::get_next_scalar<std::shared_ptr<arrow::DataType>>(ctx);
  auto scalar     = argument::get_next_scalar<bool>(ctx);
  return task::PhysicalColumn(ctx, ctx.get_next_output_arg(), arrow_type, scalar);
}

}  // namespace argument

}  // namespace legate::dataframe
