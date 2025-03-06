# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.legate cimport cpp_StoreTarget
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

    def __getitem__(self, column: int | str) -> LogicalColumn:
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

        return stack([self[n] for n in range(self.num_columns())], axis=1, out=out)

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

    def repr(self, size_t max_num_items=30) -> str:
        """Return a printable representational string

        Parameters
        ----------
        max_num_items : int
            Maximum number of items to include before items are abbreviated.

        Returns
        -------
            Printable representational string
        """
        cdef string ret = self._handle.repr(max_num_items)
        return ret.decode('UTF-8')

    def __repr__(self) -> str:
        return self.repr()
