# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import traceback
from contextlib import nullcontext
from functools import partial
from typing import Any

import polars as pl
from polars.polars import PyLazyFrame

from .dsl.translate import Translator
from .typing import NodeTraverser
from .utils.dtypes import to_polars
from .utils.versions import _ensure_polars_version

__all__: list[str] = ["collect_with_legate", "lazy_from_legate_df"]


def _create_df(df, cols, predicate, _unsure=None, chunk_if_pred=None):
    # TODO: This function should just raise an error during normal collect!
    #       Instead, we would need a `to_polars()`
    #       One could certainly hack this by using a custom object, so that the
    #       default `__call__()` raises an error.

    # TODO: Check what polars passes here (maybe not always None?)
    if predicate is not None:
        raise NotImplementedError(f"{predicate=}")
    if chunk_if_pred is not None:
        # I suspect this is chunking or so?  E.g. can be 100000
        # but could also be other predicate metadata.
        # (seems not passed in arrow bode)
        raise NotImplementedError(f"{chunk_if_pred=}")
    if _unsure is not None:
        # Unclear what this is right now.
        raise NotImplementedError(f"{_unsure=}")

    if cols is not None:
        raise NotImplementedError("currently handled during custom collect.")

    return df


def lazy_from_legate_df(df):
    schema = [(n, to_polars(df[n].dtype())) for n in df.get_column_names()]

    # Last False (within polars) changes the predicate to be passed for arrow.
    plf = PyLazyFrame.scan_from_python_function_pl_schema(
        schema, partial(_create_df, df), False
    )

    return pl.LazyFrame._from_pyldf(plf)


def collect_with_legate(query):
    _ensure_polars_version()

    q = query._ldf.visit()
    ir = _execute_with_legate(q)

    return ir.evaluate(cache={}, timer=None).table


@pl.api.register_lazyframe_namespace("legate")
class LegateOperations:
    def __init__(self, ldf: pl.LazyFrame) -> None:
        self._ldf = ldf

    def collect(self):
        return collect_with_legate(self._ldf)


def _callback(
    ir,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    *,
    device=None,
    memory_resource=None,
):
    assert with_columns is None
    assert pyarrow_predicate is None
    assert n_rows is None

    try:
        res = ir.evaluate(cache={})  # .to_polars()
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        raise

    return res


def _execute_with_legate(nt: NodeTraverser, *, config=None) -> Any:
    """
    A post optimization callback that attempts to execute the plan with legate.

    Parameters
    ----------
    nt
        NodeTraverser

    config
        GPUEngine configuration object

    Raises
    ------
    ValueError
        If the config contains unsupported keys.
    NotImplementedError
        If translation of the plan is unsupported.

    Notes
    -----
    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    # device = config.device
    # memory_resource = config.memory_resource

    # validate_config_options(config.config)

    # with nvtx.annotate(message="ConvertIR", domain="legate_df_polars"):
    with nullcontext():
        translator = Translator(nt, pl.GPUEngine(raise_on_fail=True))
        ir = translator.translate_ir()
        ir_translation_errors = translator.errors
        if len(ir_translation_errors):
            # TODO: Display these errors in user-friendly way.
            # tracked in https://github.com/rapidsai/cudf/issues/17051
            unique_errors = sorted(set(ir_translation_errors), key=str)

            def format_tb(err):
                return "\n    ".join(traceback.format_tb(err.__traceback__))

            formatted_errors = "\n\n".join(
                f"- {e.__class__.__name__}: {e}: {format_tb(e)!s}"
                for e in unique_errors
            )
            error_message = (
                "Query execution with legate not possible: unsupported operations."
                f"\nThe errors were:\n{formatted_errors}"
            )
            exception = NotImplementedError(error_message)
            raise exception
        # else:
        #     nt.set_udf(
        #         partial(
        #             _callback,
        #             ir,
        #             memory_resource=memory_resource,
        #             config_options=translator.config_options,
        #             timer=timer,
        #         )
        #     )

    return ir
