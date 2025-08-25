Polars support
==============

Legate-dataframe has support for integrating with polars.
To enable this support it is currently necessary to include::

    import legate_dataframe.ldf_polars  # noqa: F401

This will enable the ability call::

    polars_lazy_frame.legate.collect()

A minimal example may be::

    q = pl.scan_csv("mydata.csv")
    q = q.with_columns(...)  # work with q
    legate_result = q.legate.collect()

which executes the polars query to return a legate-dataframe
`~legate_dataframe.lib.core.table.LogicalTable`.

As of now, we do not hook into polars' ``collect``, nor does
legate-dataframe's ``collect`` currently fall back to polars when an
operation is unsupported.
The solution will either fully execute in a distributed manner within
legate-dataframe or an error will be raised.

If you wish to convert a ``LogicalTable`` to a ``polars.Dataframe``
you may do so via ``polars.from_arrow(legate_table.to_arrow())``.
A ``LogicalTable`` may also be converted to a polars ``LazyFrame`` via
``LogicalTable.lazy()``, however, such a ``LazyFrame`` can only be
collected via ``.legate.collect()``.

.. note::
    The exact integration path may change in the future to use
    the polars engine more like ``cudf-polars``.
    The current approach exists mainly to allow the return of a
    ``LogicalTable`` object.
