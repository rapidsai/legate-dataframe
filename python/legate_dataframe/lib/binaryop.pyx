# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport int32_t

from pylibcudf.types cimport data_type

from legate_dataframe.lib.core.column cimport LogicalColumn, cpp_LogicalColumn
from legate_dataframe.lib.core.data_type cimport as_data_type
from legate_dataframe.lib.core.scalar cimport cpp_scalar_col_from_python

from numpy.typing import DTypeLike

from legate_dataframe.lib.core.scalar import ScalarLike
from legate_dataframe.utils import _track_provenance


cdef extern from "<cudf/binaryop.hpp>" namespace "cudf":
    cpdef enum class binary_operator(int32_t):
        """Enum of binary operators, see :external:cpp:enum:`binary_operator`.
        """
        ADD,                   # operator +
        SUB,                   # operator -
        MUL,                   # operator *
        DIV,                   # operator / using common type of lhs and rhs
        TRUE_DIV,              # operator / after promoting type to floating point
        BITWISE_AND,           # operator &
        BITWISE_OR,            # operator |
        BITWISE_XOR,           # operator ^
        LOGICAL_AND,           # operator &&
        LOGICAL_OR,            # operator ||
        EQUAL,                 # operator ==
        NOT_EQUAL,             # operator !=
        LESS,                  # operator <
        GREATER,               # operator >
        LESS_EQUAL,            # operator <=
        GREATER_EQUAL,         # operator >=
        MOD,                   # operator %

        # operator //
        # integer division rounding towards negative
        # infinity if both arguments are integral;
        # floor division for floating types (using C++ type
        # promotion for mixed integral/floating arguments)
        # If different promotion semantics are required, it
        # is the responsibility of the caller to promote
        # manually before calling in to this function.
        FLOOR_DIV,
        # positive modulo operator
        # If remainder is negative, this returns
        # (remainder + divisor) % divisor else, it returns
        # (dividend % divisor)
        PMOD,
        # operator % but following Python's sign rules for negatives
        PYMOD,
        # lhs ^ rhs
        POW,
        # int ^ int, used to avoid floating point precision loss.
        # Returns 0 for negative exponents.
        INT_POW,
        # logarithm to the base
        LOG_BASE,
        # 2-argument arctangent
        ATAN2,
        # operator <<
        SHIFT_LEFT,
        # operator >>
        SHIFT_RIGHT,
        # operator >>> (from Java)
        # Logical right shift. Casts to an unsigned value before shifting.
        SHIFT_RIGHT_UNSIGNED,
        # Returns true when both operands are null; false when one is null; the
        # result of equality when both are non-null
        NULL_EQUALS,
        # Returns max of operands when both are non-null; returns the non-null
        # operand when one is null; or invalid when both are null
        NULL_MAX,
        # Returns min of operands when both are non-null; returns the non-null
        # operand when one is null; or invalid when both are null
        NULL_MIN,
        # generic binary operator to be generated with input
        # ptx code
        GENERIC_BINARY,
        # operator && with Spark rules: (null, null) is null, (null, true) is null,
        # (null, false) is false, and (valid, valid) == LOGICAL_AND(valid, valid)
        NULL_LOGICAL_AND,
        # operator || with Spark rules: (null, null) is null, (null, true) is true,
        # (null, false) is null, and (valid, valid) == LOGICAL_OR(valid, valid)
        NULL_LOGICAL_OR,
        # invalid operation
        INVALID_BINARY


cdef extern from "<legate_dataframe/binaryop.hpp>" nogil:
    cpp_LogicalColumn cpp_binary_operation "binary_operation"(
        const cpp_LogicalColumn& lhs,
        const cpp_LogicalColumn& rhs,
        binary_operator op,
        data_type output_type
    )


@_track_provenance
def binary_operation(
    lhs: LogicalColumn | ScalarLike,
    rhs: LogicalColumn | ScalarLike,
    binary_operator op,
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
        The binary operator see `binary_operator`.
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
            lhs_col._handle, rhs_col._handle, op, as_data_type(output_type)
        )
    )
