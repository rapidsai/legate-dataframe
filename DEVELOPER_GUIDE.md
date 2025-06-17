# Legate-DataFrame Developer Guide

This document serves as a guide for contributors to legate-dataframe code.

## Overview

legate-dataframe is a legate-distributed version of [libcudf](https://docs.rapids.ai/api/libcudf/stable/).

### Lexicon

This section defines terminology used within legate-dataframe.

#### Column

A column is an array of data of a single type. Along with Tables, columns are the fundamental data structures used in legate-dataframe. Most legate-dataframe algorithms operate on columns. Columns may have a validity mask representing whether each element is valid or null (invalid).

Columns come in two variants - a logical variant and a physical variant. Both variants are the legate equivalent to a libcudf [Column](https://docs.rapids.ai/api/libcudf/stable/group__column__classes.html).

##### `PhysicalColumn`
A physical column is local to single legate node and is used by legate tasks. A physical column can be zero-copied to/from a libcudf column.


##### `LogicalColumn`
A logical column is distributed between legate nodes and is the class that we expose to the client. Must of the client API in legate-dataframe takes logical columns and tables as arguments.
A LogicalColumn corresponds to legate’s LogicalArray and LogicalStore.

Note that `LogicalColumn` can be marked as scalar.  By default, this information
is largely ignored, but can be used to specialize for scalars, see e.g. binary operations.

#### Table

A table is a collection of columns with equal number of elements. Tables come in two variants - a logical variant and a physical variant. Both variants are the legate equivalent to a libcudf [Table](https://docs.rapids.ai/api/libcudf/stable/group__table__classes).

##### `PhysicalTable`
A physical table is local to single legate node and is used by legate tasks. A physical table can be zero-copied to/from a libcudf table.

##### `LogicalTable`
A logical column is distributed between legate nodes and is the class that the client sees. Must of the client API in legate-dataframe takes logical columns and tables as arguments.

A LogicalTable corresponds to legate’s LogicalArray and LogicalStore.

#### Element

An individual data item within a column. Also known as a row.

#### Scalar

A type representing a single element of a data type. In `legate-dataframe`
a scalar is represented by a `LogicalColumn` for which `LogicalColumn.is_scalar()`
returns true.

## Task Implementation
Currently, we only implement GPU task variants.

### Context

To reduce boilerplate code, standardize the retrieval of task arguments, and ensure correct use of CUDA streams and allocations, each task creates a `TaskContext` instance as its very first thing. Task arguments such as `PhysicalTable`, `PhysicalColumn`, and scalars can then be retrieved using this context instance.

In the following code snippets, we have a task that retrieve its arguments using `TaskContext`. Notice, the order of the argument retrieval **must** match the order the arguments are added to the task.

```c++
namespace legate::dataframe {  // The public namespace
namespace task { // The private namespace for task specific code

// The private task function, which isn't exposed in the header
class UnaryOpTask : public Task<UnaryOpTask, OpCode::UnaryOp> {
 public:
  static void gpu_variant(legate::TaskContext context)
  {
    TaskContext ctx{context};
    auto op                           = argument::get_next_scalar<cudf::unary_operator>(ctx);
    const auto input                  = argument::get_next_input<PhysicalColumn>(ctx);
    auto output                       = argument::get_next_output<PhysicalColumn>(ctx);
    cudf::column_view col             = input.column_view();
    std::unique_ptr<cudf::column> ret = cudf::unary_operation(col, op, ctx.stream(), ctx.mr());
    output.move_into(std::move(ret));
  }
};

}  // namespace task

// The public API exposed to the client
LogicalColumn unary_operation(const LogicalColumn& col, cudf::unary_operator op)
{
  auto runtime          = legate::Runtime::get_runtime();
  auto ret              = LogicalColumn::empty_like(col);
  legate::AutoTask task = runtime->create_task(get_library(), task::OpCode::UnaryOp);
  argument::add_next_scalar(task, static_cast<std::underlying_type_t<cudf::unary_operator>>(op));
  argument::add_next_input(task, col);
  argument::add_next_output(task, ret);
  runtime->submit(std::move(task));
  return ret;
}
}  // namespace legate::dataframe
```

Notice, it is possible to mix the task argument API from legate-dataframe and legate.core but it requires that the legate-dataframe API is used continuously either before or after legate.core API.
To do this, use ``TaskContext.get_task_argument_indices`` or initialize it with the correct offsets.



### CUDA Stream and Memory Allocation

Always use the CUDA stream from `stream()` and `mr()` RMM resource from `TaskContext`. This is because Legate may run multiple tasks on the same GPU and calls to CUDA functions such as `cudaMalloc()` might block **all** CUDA kernels on the same device. By using `TaskContext::stream()` and `TaskContext::mr()` exclusively, we use [`Legion::DeferredBuffer`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L3509-L3609):
> We use Legion `DeferredBuffer`, whose lifetime is not connected with the CUDA stream(s) used to launch kernels. The buffer is allocated immediately at the point when `create_buffer` is called, whereas the kernel that uses it is placed on a stream, and may run at a later point. Normally a `DeferredBuffer` is deallocated automatically by Legion once all the kernels launched in the task are complete. However, a `DeferredBuffer` can also be deallocated immediately using `destroy()`, which is useful for operations that want to deallocate intermediate memory as soon as possible. This deallocation is not synchronized with the task stream, i.e. it may happen before a kernel which uses the buffer has actually completed. This is safe as long as we use the same stream on all GPU tasks running on the same device, because then all the actual uses of the buffer are done in order on the one stream. It is important that all library CUDA code uses `ctx.stream()`, and all CUDA operations (including library calls) are enqueued on that stream exclusively. This analysis additionally assumes that no code outside of Legate is concurrently allocating from the eager pool, and that it's OK for kernels to access a buffer even after it's technically been deallocated.
Always use the CUDA stream from `context->get_task_stream()` and a `TaskMemoryResource` local to the task. This is because Legate may run multiple tasks on the same GPU and calls to CUDA functions such as `cudaMalloc()` might block **all** CUDA kernels on the same device.

Notice, if cuDF's public API doesn't accept a stream argument, we use cuDF's internal API.

### CUDA Error Checking

Use the `LEGATE_CHECK_CUDA` macro to check for the successful completion of CUDA runtime API functions
(although this will need to be replaced with our own version eventually).
If the CUDA API return value is not `cudaSuccess`, the macro prints a description of the CUDA error code and exit the process.

Example:

```c++
LEGATE_CHECK_CUDA( cudaMemcpy(&dst, &src, num_bytes) );
```
