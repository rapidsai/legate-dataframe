# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string

from pyarrow.lib cimport CDataType

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_arrow_data_type
from legate_dataframe.lib.core.scalar cimport cpp_scalar_col_from_python

from numpy.typing import DTypeLike

from legate_dataframe.utils import _track_provenance


cdef extern from "<legate_dataframe/binaryop.hpp>" nogil:
    cpp_LogicalColumn cpp_binary_operation "binary_operation"(
        const cpp_LogicalColumn& lhs,
        const cpp_LogicalColumn& rhs,
        string op,
        shared_ptr[CDataType] output_type
    )


@_track_provenance
def binary_operation(
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
    op: str,
    output_type: DTypeLike
) -> LogicalColumn:
    """Performs a binary operation between two columns or a column and a scalar.

    The output contains the result of ``op(lhs[i], rhs[i])`` for all
    ``0 <= i < lhs.size()`` where ``lhs[i]`` or ``rhs[i]`` (but not both) can be
    replaced with a scalar value.

    Regardless of the operator, the validity of the output value is the
    logical AND of the validity of the two operands except for NullMin and
    NullMax (logical OR).

    Parameters
    ----------
    lhs
        The left operand
    lhs
        The right operand
    op
        String for arrow compute function e.g. "add", "multiply"
    output_type
        The desired data type of the output column

    Returns
    -------
        Output column of `output_type` type containing the result of the binary
        operation

    Raises
    ------
    ValueError
        if `lhs` and `rhs` are both scalars
    RuntimeError
        if `lhs` and `rhs` are different sizes
    RuntimeError
        if `output_type` dtype isn't boolean for comparison and logical operations.
    RuntimeError
        if `output_type` dtype isn't fixed-width
    RuntimeError
        if the operation is not supported for the types of `lhs` and `rhs`

    """
    cdef LogicalColumn lhs_col
    cdef LogicalColumn rhs_col
    # If an input is not a column, assume it is scalar:
    if isinstance(lhs, LogicalColumn):
        lhs_col = <LogicalColumn>lhs
    else:
        lhs_col = cpp_scalar_col_from_python(lhs)

    if isinstance(rhs, LogicalColumn):
        rhs_col = <LogicalColumn>rhs
    else:
        rhs_col = cpp_scalar_col_from_python(rhs)

    return LogicalColumn.from_handle(
        cpp_binary_operation(
            lhs_col._handle, rhs_col._handle, op.encode('utf-8'),
            as_arrow_data_type(output_type)
        )
    )
