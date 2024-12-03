# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column as cudfColumn
from pylibcudf.libcudf.column.column cimport column

from legate_dataframe.lib.core.legate_task cimport get_auto_task_handle
from legate_dataframe.lib.core.logical_array cimport cpp_LogicalArray

from typing import Any

from cudf._typing import DtypeObj
from legate.core import AutoTask, Field, LogicalArray

from legate_dataframe.lib.core.data_type cimport cpp_cudf_type_to_cudf_dtype

from legate_dataframe.utils import get_logical_array


cdef class LogicalColumn:
    """Logical column distributed between legate nodes

    Underlying a logical column is a logical array. The column doesn't own the array,
    a logical array can be part of multiple columns.
    """

    def __init__(self, obj: Any):
        """Create a new logical column using the legate data interface

        The object must expose a single logical array.

        Parameters
        ----------
        obj
            Objects that exposes `__legate_data_interface__` interface.
        """
        array = get_logical_array(obj)
        assert type(array) is LogicalArray
        cdef uintptr_t raw_handle = array.raw_handle
        self._handle = cpp_LogicalColumn(dereference(<cpp_LogicalArray*> raw_handle))

    @staticmethod
    cdef LogicalColumn from_handle(cpp_LogicalColumn handle):
        """Create a new logical column from a C++ handle

        Parameters
        ----------
        handle
            C++ handle of a LogicalColumn.

        Returns
        -------
            Logical column representing the existing C++ LogicalColumn
        """
        cdef LogicalColumn ret = LogicalColumn.__new__(LogicalColumn)
        ret._handle = handle
        return ret

    @staticmethod
    def from_cudf(cudfColumn col) -> LogicalColumn:
        """Create a logical column from a local cudf column.

        This call blocks the client's control flow and scatter
        the data to all legate nodes.

        Parameters
        ----------
        col
            cudf column

        Returns
        -------
            New logical column
        """
        return LogicalColumn.from_handle(cpp_LogicalColumn(col.view()))

    @staticmethod
    def empty_like_logical_column(LogicalColumn col) -> LogicalColumn:
        """Create a new unbounded column from an existing column.

        Parameters
        ----------
        other : LogicalColumn
            The prototype column.

        Returns
        -------
            The new unbounded column with the type and nullable equal `other`
        """
        return LogicalColumn.from_handle(cpp_LogicalColumn.empty_like(col._handle))

    def num_rows(self) -> int:
        """Returns the number of rows.

        Returns
        -------
            The number of rows

        Raises
        ------
        RuntimeError
            if column is unbound
        """
        return self._handle.num_rows()

    def dtype(self) -> DtypeObj:
        """Returns the cudf data type of the row elements

        Returns
        -------
            The cudf data type
        """
        return cpp_cudf_type_to_cudf_dtype(self._handle.cudf_type())

    def get_logical_array(self) -> LogicalArray:
        """Return the underlying logical array

        Returns
        -------
            The underlying logical array
        """
        cdef cpp_LogicalArray ary = self._handle.get_logical_array()
        return LogicalArray.from_raw_handle(<uintptr_t> &ary)

    @property
    def __legate_data_interface__(self):
        array = self.get_logical_array()
        return {
            "version": 1,
            "data": {Field("LogicalColumn", array.type): array},
        }

    def to_cudf(self) -> cudfColumn:
        """Copy the logical column into a local cudf column

        This call blocks the client's control flow and fetches the data for the
        whole column to the current node.

        Returns
        -------
            A cudf column, which own the data.
        """
        cdef unique_ptr[column] col = self._handle.get_cudf()
        return cudfColumn.from_unique_ptr(move(col))

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

    def add_as_next_task_input(self, task: AutoTask) -> None:
        """Add a logical column to the next input task argument

        This should match a call to `get_next_input<PhysicalColumn>()` by a legate task.

        .. note::
            The order of "add_next_*" calls must match the order of the
            corresponding "get_next_*" calls.

        Parameters
        ----------
        task
            The legate task to add the argument.
        """
        cdef cpp_AutoTask *cpp_task = get_auto_task_handle(task)
        cpp_add_next_input(dereference(cpp_task), self._handle)

    def add_as_next_task_output(self, task: AutoTask) -> None:
        """Add a logical column to the next output task argument

        This should match a call to `get_next_input<PhysicalColumn>()` by a legate task.

        .. note::
            The order of "add_next_*" calls must match the order of the
            corresponding "get_next_*" calls.

        Parameters
        ----------
            The legate task to add the argument.
        """
        cdef cpp_AutoTask *cpp_task = get_auto_task_handle(task)
        cpp_add_next_output(dereference(cpp_task), self._handle)
