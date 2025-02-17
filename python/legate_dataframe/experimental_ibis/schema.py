# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import IbisError

import cudf  # for max precisions
from pylibcudf.types import DataType, TypeId

from legate_dataframe import LogicalTable

_to_plc_type_id = {
    dt.Boolean: TypeId.BOOL8,
    # dt.Null: ,
    dt.String: TypeId.STRING,
    # dt.Binary: ,
    # Note: Scalars may end up using seconds here:
    dt.Date: TypeId.TIMESTAMP_DAYS,
    # dt.Time: TypeId.TIMESTAMP_NANOSECONDS,  # TODO: what is right for time?
    dt.Int8: TypeId.INT8,
    dt.Int16: TypeId.INT16,
    dt.Int32: TypeId.INT32,
    dt.Int64: TypeId.INT64,
    dt.UInt8: TypeId.UINT8,
    dt.UInt16: TypeId.UINT16,
    dt.UInt32: TypeId.UINT32,
    dt.UInt64: TypeId.UINT64,
    dt.Float32: TypeId.FLOAT32,
    dt.Float64: TypeId.FLOAT64,
}


_to_plc_type = {i_t: DataType(plc_tn) for i_t, plc_tn in _to_plc_type_id.items()}

_from_legate_df_type = {v: k for k, v in _to_plc_type.items()}


def to_plc_type(ibis_type):
    if not ibis_type.nullable:
        # Columns may not have a mask as an optimization, but logically we do
        # never enforce this.  So it could make sense to allow and use this.
        raise IbisError("non-nullable types are not supported by Legate.")
    if type(ibis_type) == dt.Decimal:
        if ibis_type.precision <= cudf.Decimal32Dtype.MAX_PRECISION:
            return DataType(TypeId.DECIMAL32, -ibis_type.scale)
        elif ibis_type.precision <= cudf.Decimal64Dtype.MAX_PRECISION:
            return DataType(TypeId.DECIMAL64, -ibis_type.scale)
        elif ibis_type.precision <= cudf.Decimal128Dtype.MAX_PRECISION:
            return DataType(TypeId.DECIMAL128, -ibis_type.scale)
        else:
            raise NotImplementedError("Unsupported decimal precision")

    return _to_plc_type[type(ibis_type)]


def to_ibis_type(plc_type):
    if plc_type.id() == TypeId.DECIMAL32:
        return dt.Decimal(int(cudf.Decimal32Dtype.MAX_PRECISION), -plc_type.scale())
    elif plc_type.id() == TypeId.DECIMAL64:
        return dt.Decimal(int(cudf.Decimal64Dtype.MAX_PRECISION), -plc_type.scale())
    elif plc_type.id() == TypeId.DECIMAL128:
        return dt.Decimal(int(cudf.Decimal128Dtype.MAX_PRECISION), -plc_type.scale())

    return _from_legate_df_type[plc_type]


# TODO: Do I need the full Schema/Datamapper? (likely reasonable abstractions to add)
def infer_schema_from_logical_table(ldf: LogicalTable) -> sch.Schema:
    info = []

    for col_name in ldf.get_column_names():
        col = ldf[col_name]
        ibis_dtype = to_ibis_type(col.dtype())

        info.append((col_name, ibis_dtype))

    return sch.Schema.from_tuples(info)


def get_names_dtypes_from_schema(
    schema: sch.Schema,
) -> Tuple[List[str], List[DataType]]:
    names = schema.names
    types = [_to_plc_type[type(t)] for t in schema.types]
    return names, types
