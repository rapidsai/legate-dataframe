# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import pyarrow as pa
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.legate cimport cpp_StoreTarget, from_python_slice
from legate_dataframe.lib.core.table cimport cpp_LogicalTable

from typing import Iterable

import cudf


cdef cpp_LogicalColumn get_logical_column_handle(col: LogicalColumn):
    cdef LogicalColumn cpp_column = col
    return cpp_column._handle


cdef class LogicalTable:
    """Collection of logical columns

    The order of the collection of columns is preserved. Use `.get_column`
    and `.get_columns` to access individual columns.

    Unlike libcudf, the columns in a `LogicalTable` have names, which makes it possible
    to retrieve columns by name using `.get_column()`. Additionally, when reading and
    writing tables to/from files, the column names are read and written automatically.

    Notice, the table doesn't *own* the columns, a column can be in multiple tables.
    """

    def __init__(self, columns: Iterable[LogicalColumn], column_names: Iterable[str]):
        """Create a table from a vector of columns

        Parameters
        ----------
        columns
            The columns to be part of the logical table.
        column_names
            Column names given in the same order as `columns`.
        """
        cdef vector[cpp_LogicalColumn] handles
        for col in columns:
            handles.push_back(get_logical_column_handle(col))

        cdef vector[string] col_names
        for name in column_names:
            col_names.push_back(name.encode('UTF-8'))

        self._handle = cpp_LogicalTable(move(handles), col_names)

    @staticmethod
    cdef LogicalTable from_handle(cpp_LogicalTable handle):
        """Create a new logical table from a C++ handle.

        Parameters
        ----------
        handle
            C++ handle of a LogicalTable.

        Returns
        -------
            Logical table representing the existing C++ LogicalColumn
        """
        cdef LogicalTable ret = LogicalTable.__new__(LogicalTable)
        ret._handle = handle
        return ret

    @staticmethod
    def from_cudf(df: cudf.DataFrame) -> LogicalTable:
        """Create a logical table from a local cudf dataframe

        This call blocks the client's control flow and scatter
        the data to all legate nodes.

        Parameters
        ----------
        df : cudf.DataFrame
            cudf dataframe

        Returns
        -------
            New logical table
        """
        return LogicalTable(
            columns=(LogicalColumn.from_cudf(c) for c in df._columns),
            column_names=df.columns
        )

    @staticmethod
    def from_arrow(table: pa.Table) -> LogicalTable:
        """Create a logical table from a local arrow table

        This call blocks the client's control flow and scatter
        the data to all legate nodes.

        Parameters
        ----------
        table : pyarrow.Table
            Arrow table

        Returns
        -------
            New logical table
        """
        columns = [LogicalColumn.from_arrow(a.combine_chunks()) for a in table.columns]
        return LogicalTable(
            columns=columns,
            column_names=table.column_names,
        )

    def lazy(self):
        # Import here to avoid hard polars dependency
        from legate_dataframe.ldf_polars import lazy_from_legate_df

        return lazy_from_legate_df(self)

    def num_columns(self) -> int:
        """Returns the number of columns

        Returns
        -------
            The number of columns
        """
        return self._handle.num_columns()

    def num_rows(self) -> int:
        """Returns the number of rows

        Returns
        -------
        int
            The number of rows

        Raises
        ------
        RuntimeError
            if table is unbound
        """
        return self._handle.num_rows()

    cdef LogicalColumn get_column_by_index(self, size_t idx):
        """Returns a reference to the specified column

        Parameters
        ----------
        idx : int
            Index of the desired column

        Returns
        -------
            The desired column

        Raises
        ------
        IndexError
            If ``idx`` is out of the range ``[0, num_columns)``
        TypeError
            If `column` isn't an integer
        OverflowError
            If `column` is a negative integer
        """
        return LogicalColumn.from_handle(self._handle.get_column(idx))

    def get_column(self, column: int | str) -> LogicalColumn:
        """Returns a reference to the specified column

        Parameters
        ----------
        column : int or str
            Index or name of the desired column

        Returns
        -------
            The desired column

        Raises
        ------
        IndexError
            If `column` doesn't exist
        TypeError
            If `column` isn't a string or integer
        OverflowError
            If `column` is a negative integer
        """
        if isinstance(column, str):
            return LogicalColumn.from_handle(
                self._handle.get_column(<string> column.encode('UTF-8'))
            )
        return self.get_column_by_index(column)

    def select(self, columns) -> LogicalTable:
        """Select a subset of columns from this table.

        Similar to ``table[columns]`` but accepts any iterable.

        Parameter
        ---------
        columns
            Iterable of column names or indices.

        Returns
        -------
        table
            A table with only the selected columns.
        """
        cdef vector[string] col_names
        cdef vector[size_t] col_indices

        columns = list(columns)
        if len(columns) > 0 and isinstance(columns[0], str):
            for name in columns:
                # Check for string, error is an AttributeError otherwise.
                if not isinstance(name, str):
                    raise TypeError(
                        "column names must be an iterable of str or int and not mixed")
                col_names.push_back(name.encode('UTF-8'))
            return LogicalTable.from_handle(self._handle.select(col_names))
        else:
            col_indices = columns
            return LogicalTable.from_handle(self._handle.select(col_indices))

    def __getitem__(self, column):
        """Returns a reference to the specified column

        Parameters
        ----------
        column : int, str, or list of them
            Index or name of the desired column.  If a list the return
            will be a new table with the selected columns.

        Returns
        -------
            The desired column or a table if a list.

        Raises
        ------
        IndexError
            If `column` doesn't exist
        TypeError
            If `column` isn't a string or integer, or list of these.
        OverflowError
            If a negative integer is encountered.
        """
        if isinstance(column, list):
            return self.select(column)
        return self.get_column(column)

    def get_column_names(self) -> List[str]:
        """Returns a list of the column names order by column indices

        Returns
        -------
        list of str
            A list of the column names
        """
        cdef vector[string] names = self._handle.get_column_name_vector()
        ret = []
        for i in range(names.size()):
            ret.append(names.at(i).decode('UTF-8'))
        return ret

    def offload_to(self, cpp_StoreTarget target_mem):
        """Offload the underlying data to the specified memory.

        This method offloads the underlying data to the specified target memory.
        The purpose of this is to free up GPU memory resources.
        See :external:cpp:func:`legate::LogicalArray::offload_to` for more
        information.

        Parameters
        ----------
        target_mem : legate.core.StoreTarget
            The memory kind to offload to. To offload to the CPU use
            ``legate.core.StoreTarget.SYSMEM``.
        """
        self._handle.offload_to(target_mem)

    def slice(self, slice_):
        """Slice the table by rows, this is the same as slicing all columns
        individually.

        Parameters
        ----------
        slice_ :
            Python slice. The return will be a view in the original data.

        Returns
        -------
            The sliced logical table as a view.
        """
        return LogicalTable.from_handle(
            self._handle.slice(from_python_slice(slice_))
        )

    def to_array(self, *, out=None):
        """Convert the table or a set of columns to a cupynumeric array.

        The returned array is always a copy.

        Parameters
        ----------
        out
            If given an output cupynumeric array.

        Returns
        -------
        array
            A cupynumeric array of shape ``(num_rows, num_cols)``.
        """
        from cupynumeric import stack

        return stack(
            [self[n].to_array() for n in range(self.num_columns())], axis=1, out=out
        )

    def to_cudf(self) -> cudf.DataFrame:
        """Copy the logical table into a local cudf table

        This call blocks the client's control flow and fetches the data for the
        whole table to the current node.

        Returns
        -------
        cudf.DataFrame
            A local cudf dataframe copy.
        """
        ret = cudf.DataFrame()
        for i, name in enumerate(self.get_column_names()):
            ret[name] = self.get_column(i).to_cudf()
        return ret

    def to_arrow(self) -> pa.Table:
        ret = pa.table(
            [self.get_column(i).to_arrow() for i in range(self.num_columns())],
            names=self.get_column_names(),
        )
        return ret

    def __repr__(self) -> str:
        return self.to_arrow().__repr__()
