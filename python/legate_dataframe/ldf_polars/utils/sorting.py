# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sorting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def sort_order(
    descending: Sequence[bool], *, nulls_last: Sequence[bool], num_keys: int
) -> tuple[list[bool], bool]:
    """
    Produce sort order arguments.

    Parameters
    ----------
    descending
        List indicating order for each column
    nulls_last
        Should nulls sort last or first?
    num_keys
        Number of sort keys

    Returns
    -------
    tuple of column_order and null_precedence
    suitable for passing to sort routines
    """
    # Mimicking polars broadcast handling of descending
    if num_keys > (n := len(descending)) and n == 1:
        descending = [descending[0]] * num_keys
    sort_ascending = [not d for d in descending]

    if not all(x == nulls_last[0] for x in nulls_last):
        raise ValueError(
            "All nulls_last values must be the same for this implementation"
        )

    return sort_ascending, nulls_last[0]
