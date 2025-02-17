# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap

import cudf
import cupy
import ibis

from legate_dataframe import LogicalTable
from legate_dataframe.experimental_ibis import LegateBackend
from legate_dataframe.testing import assert_frame_equal


def test_ibis_basic_read_csv_and_binary(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text(
        textwrap.dedent(
            """\
        a,b,c
        1,2,3
        2,3,4
        """
        )
    )

    schema = ibis.schema({"a": "int64", "b": "float64", "c": "int64"})
    con = LegateBackend().connect()
    # NOTE: As of writing, require schema.  In principle should infer schema
    # and not actually read the full file immediately (i.e. make lazy).
    table = con.read_csv(file, schema=schema)

    sum_ab = (table.a + table.b).name("sum_ab")

    res = sum_ab.execute()  # A columnar (pandas) result.
    assert res.name == "sum_ab"  # pandas series has a name and we honor it
    # Rename res, just for `assert_frame_equal`.
    assert_frame_equal(res.rename("data"), cudf.Series([3.0, 5.0]))


def test_ibis_basic_read_csv_and_groupby_agg(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text(
        textwrap.dedent(
            """\
        a,b,c
        1,2,3
        2,3,4
        1,3,4
        """
        )
    )

    schema = ibis.schema({"a": "int64", "b": "float64", "c": "int64"})

    con = LegateBackend().connect()
    # NOTE: As of writing, require schema.  In principle should infer schema
    # and not actually read the full file immediately (i.e. make lazy).
    table = con.read_csv(file, schema=schema)

    res = table.group_by(["a"]).aggregate(
        sum_ab=(table.a + table.b).sum(),
        max_ab=(table.a + table.b).max(),
        sum_c=table.c.sum(),
    )

    res = con.to_legate(res)  # A legate result
    expected = cudf.DataFrame(
        {"a": [1, 2], "sum_ab": [7.0, 5.0], "max_ab": [4.0, 5.0], "sum_c": [7, 4]}
    )

    assert_frame_equal(res, expected)


def test_ibis_basic_join_chain():
    df1 = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "payload_a": cupy.arange(5)})
    df2 = cudf.DataFrame({"b": [1, 1, 2, 2, 5, 6], "c": cupy.arange(6) * -1})
    df3 = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "c": cupy.arange(6) * -2,
            "payload_c": cupy.arange(6) - 3,
        }
    )

    con = LegateBackend().connect(
        {
            "df1": LogicalTable.from_cudf(df1),
            "df2": LogicalTable.from_cudf(df2),
            "df3": LogicalTable.from_cudf(df3),
        }
    )

    ibis1 = con.table("df1")
    ibis2 = con.table("df2")
    ibis3 = con.table("df3")
    # First join based on predicates, then based on names from both original ones:
    expr = ibis1.join(ibis2, ibis1.a == ibis2.b).join(ibis3, ["a", "c"])

    res = expr.execute()  # pandas result
    expected = df1.merge(df2, left_on="a", right_on="b").merge(df3, on=["a", "c"])
    assert_frame_equal(res, expected)


def test_ibis_basic_join_same_colnames():
    df1 = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": cupy.arange(5)})
    df2 = cudf.DataFrame({"a": [1, 1, 2, 2, 5, 6], "b": cupy.arange(6) * -1})

    con = LegateBackend().connect(
        {
            "df1": LogicalTable.from_cudf(df1),
        }
    )

    ibis1 = con.table("df1")
    # Same as df2 above, but use memtable API:
    ibis2 = ibis.memtable({"a": [1, 1, 2, 2, 5, 6], "b": (cupy.arange(6) * -1).get()})
    # First join based on predicates, then based on names from both original ones:
    expr = ibis1.join(ibis2, ibis1.a == ibis2.a)

    res = expr.execute()  # pandas result
    expected = df1.merge(df2, on="a", suffixes=("", "_right"))
    assert_frame_equal(res, expected)


def test_ibis_basic_scalar_and_mutate():
    df1 = cudf.DataFrame({"a": [1, -2, 3]})

    con = LegateBackend().connect({"df1": LogicalTable.from_cudf(df1)})

    t = con.table("df1")
    t = t.mutate(b=t.a + 3.0 + t.a.abs(), c=t.a.abs())
    res = t.execute()  # pandas result

    expected = cudf.DataFrame({"a": [1, -2, 3], "b": [5.0, 3.0, 9.0], "c": [1, 2, 3]})
    assert_frame_equal(res, expected)


def test_ibis_basic_filter():
    df1 = cudf.DataFrame({
        "a": [-1, -2, -3, -4, 5, 6, 7, 8],
        "mask": [True, None, False, True, False, None, False, True]
    })

    con = LegateBackend().connect({"df1": LogicalTable.from_cudf(df1)})

    t = con.table("df1")
    t = t.filter(t.mask, t.a > 0)
    res = t.execute()  # pandas result

    expected = df1[df1["mask"] & (df1["a"] > 0)]
    assert_frame_equal(res, expected)
