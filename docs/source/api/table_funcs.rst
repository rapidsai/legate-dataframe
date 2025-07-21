Table functions
===============

.. autofunction::
    legate_dataframe.lib.stream_compaction.apply_boolean_mask

.. autofunction::
    legate_dataframe.lib.groupby_aggregation.groupby_aggregation

.. autofunction::
    legate_dataframe.lib.join.join

.. autofunction::
    legate_dataframe.lib.sort.sort


Related options/enums
---------------------

.. py:data:: legate_dataframe.lib.groupby_aggregation.AggregationKind

    Aggregation kind as defined by :external:cpp:enum:`cudf::aggregation::Kind`.

    ..
        NOTE: Aggregation kind is coming from pylibcudf, but as of writing
        it had no good documentation.

.. autodata:: legate_dataframe.lib.join.BroadcastInput
    :no-value:

.. autodata:: legate_dataframe.lib.join.JoinType
    :no-value:

.. autodata:: legate_dataframe.lib.join.null_equality
    :no-value:
