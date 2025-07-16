# Legate-dataframe: a scalable dataframe library

A prototype of a legate-enabled version of [libcudf](https://docs.rapids.ai/api/libcudf/stable/).
This is **not** a drop-in replacement of [Pandas](https://pandas.pydata.org/), instead it follows the more low-level API of libcudf.

In the future, we plan to introduce a high-level pure Python package that implements all the nice-to-have features known from Pandas using the low-level API's primitives.

[Python API and further documentation](https://rapidsai.github.io/legate-dataframe/).

## Install

You can install `legate-dataframe` packages from the [conda legate channel](https://anaconda.org/legate/)
using
```bash
conda install -c legate -c rapidsai -c conda-forge legate-dataframe
```
To include development releases add the `legate/label/experimental` channel.

## Build

Legate-dataframe uses the Legate C++ API from Legate-core and cuPyNumeric.
cuPyNumeric is only used in Python tests and examples so it isn't strictly necessary.

The current tested versions are legate and cuPyNumeric 24.11 release available from
the [conda legate channel](https://anaconda.org/legate/).

### Legate-dataframe

First we clone `legate-dataframe` and install the dependencies:
```
git clone https://github.com/rapidsai/legate-dataframe.git
cd legate-dataframe
mamba env update --name legate-dev --file conda/environments/all_cuda-124_arch-x86_64.yaml
```
Then we can build, install, and test the project:
```
./build.sh
./build.sh test
```

## Feature Status
| Feature                              | Status                 | Limitations
|--------------------------------------|:----------------------:|----------------------------------|
| Copy to/from cuDF DataFrame          | :white_check_mark:     |                                  |
| Parquet read & write                 | :white_check_mark:     |                                  |
| CSV read & write                     | :white_check_mark:     |                                  |
| Zero-copy to/from cuPyNumeric arrays | :white_check_mark:     |                                  |
| Hash based inner join                | :white_check_mark:     |                                  |
| Hash based left join                 | :white_check_mark:     |                                  |
| Hash based full/outer join           | :white_check_mark:     |                                  |
| GroupBy Aggregation                  | :white_check_mark:     | Basic aggs. like SUM and NUNIQUE |
| Numeric data types                   | :white_check_mark:     |                                  |
| Datetime data types                  | :white_check_mark:     |                                  |
| String data types                    | :white_check_mark:     |                                  |
| Null masked columns                  | :white_check_mark:     |                                  |

## Example

### Python
```python
import tempfile
import cudf
import cupynumeric
from legate.core import get_legate_runtime
from legate_dataframe import LogicalColumn, LogicalTable
from legate_dataframe.lib.parquet import parquet_read, parquet_write

def main(tmpdir):
    # Let's start by creating a logical table from a cuDF dataframe
    # This takes a local dataframe and distribute it between Legate nodes
    df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [-1, -2, -3, -4]})
    tbl1 = LogicalTable.from_cudf(df)

    # We can write the logical table to disk using the Parquet file format.
    # The table is written into multiple files, one file per partition:
    #      /tmpdir/
    #          ├── part-0.parquet
    #          ├── part-1.parquet
    #          ├── part-2.parquet
    #          └── ...
    parquet_write(tbl1, path=tmpdir)

    # NB: since Legate execute tasks lazily, we issue a blocking fence
    #     in order to wait until all files has been written to disk.
    get_legate_runtime().issue_execution_fence(block=True)

    # Then we can read the parquet files back into a logical table. We
    # provide a Glob string that reference all the parquet files that
    # should go into the logical table.
    tbl2 = parquet_read(f"{tmpdir}/*.parquet")

    # LogicalColumn implements the `__legate_data_interface__` interface,
    # which makes it possible for other Legate libraries, such as cuPyNumeric,
    # to operate on columns seamlessly.
    ary = cupynumeric.add(tbl1["a"], tbl2["b"])
    assert ary.sum() == 0
    ary[:] = [4, 3, 2, 1]

    # We can create a new logical column from any 1-D array like object that
    # exposes the `__legate_data_interface__` interface.
    col = LogicalColumn(ary)

    # We can create a new logical table from existing logical columns.
    LogicalTable(columns=(col, tbl2["b"]), column_names=["a", "b"])

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(tmpdir)
        # Since Legate execute tasks lazily, we issue a blocking fence here
        # to make sure all task has finished before `tmpdir` is removed.
        get_legate_runtime().issue_execution_fence(block=True)
```

### C++
```c++
#include <filesystem>
#include <legate.h>

#include <legate_dataframe/core/column.hpp>
#include <legate_dataframe/core/table.hpp>
#include <legate_dataframe/parquet.hpp>
#include <legate_dataframe/unaryop.hpp>
#include <legate_dataframe/utils.hpp>

int main(int argc, char** argv)
{
  // First we initialize Legate use either `legate` or `LEGATE_CONFIG` to customize launch
  legate::start();

  // Then let's create a new logical column
  legate::dataframe::LogicalColumn col_a = legate::dataframe::sequence(20, -10);

  // Compute the absolute value of each row in `col_a`
  legate::dataframe::LogicalColumn col_b = unary_operation(col_a, cudf::unary_operator::ABS);

  // Create a new logical table that contains the two existing columns (zero-copy)
  legate::dataframe::LogicalTable tbl_a{{col_a, col_a}};

  // We can write the logical table to disk using the Parquet file format.
  // The table is written into multiple files, one file per partition:
  //      /tmpdir/
  //          ├── part-0.parquet
  //          ├── part-1.parquet
  //          ├── part-2.parquet
  //          └── ...
  legate::dataframe::parquet_write(tbl_a, "./my_parquet_file");

  // NB: since Legate execute tasks lazily, we issue a blocking fence
  //     in order to wait until all files has been written to disk.
  legate::Runtime::get_runtime()->issue_execution_fence(true);

  // Then we can read the parquet files back into a logical table. We
  // provide a Glob string that reference all the parquet files that
  // should go into the logical table.
  auto files = legate::dataframe::parse_glob("./my_parquet_file/*.parquet");
  auto tbl_b = legate::dataframe::parquet_read(files);

  // Clean up
  std::filesystem::remove_all("./my_parquet_file");
  return 0;
}
```

## Contributing

Please see our [our guide](CONTRIBUTING.md) and the [developer guide](DEVELOPER_GUIDE.md).
