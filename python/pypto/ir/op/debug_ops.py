# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Debug operations for PyPTO IR."""

from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, ScalarType, Span

from ..utils import _get_span_or_capture, _normalize_expr

_PRINTF_FLAGS = set("-+ #0")
_INTEGER_CONVERSIONS = {"d", "u", "x"}
_FLOAT_CONVERSIONS = {"f"}
_SUPPORTED_CONVERSIONS = _INTEGER_CONVERSIONS | _FLOAT_CONVERSIONS
_INTEGER_DTYPES = tuple(
    dtype
    for dtype_name in (
        "INT4",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT4",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    )
    if hasattr(DataType, dtype_name)
    for dtype in (getattr(DataType, dtype_name),)
)


def _is_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsInt"):
        return dtype == DataType.INDEX or dtype.IsInt()
    return dtype == DataType.INDEX or dtype in _INTEGER_DTYPES


def _scan_printf_format(format_str: str) -> list[str]:
    specs: list[str] = []
    i = 0
    while i < len(format_str):
        if format_str[i] != "%":
            i += 1
            continue
        if i + 1 < len(format_str) and format_str[i + 1] == "%":
            i += 2
            continue

        j = i + 1
        while j < len(format_str) and format_str[j] in _PRINTF_FLAGS:
            j += 1
        while j < len(format_str) and format_str[j].isdigit():
            j += 1
        if j < len(format_str) and format_str[j] == ".":
            j += 1
            if j >= len(format_str) or not format_str[j].isdigit():
                raise ValueError("printf precision must be followed by digits")
            while j < len(format_str) and format_str[j].isdigit():
                j += 1
        if j >= len(format_str):
            raise ValueError("printf format string ends with an incomplete conversion")

        conversion = format_str[j]
        if conversion not in _SUPPORTED_CONVERSIONS:
            raise ValueError(f"printf does not support conversion '%{conversion}'")

        specs.append(format_str[i : j + 1])
        i = j + 1

    if not specs:
        raise ValueError("printf format string must contain at least one supported conversion")
    return specs


def printf_(format_str: str, *args: int | float | Expr, span: Span | None = None) -> Call:
    """Print scalar values using a compile-time format string."""
    actual_span = _get_span_or_capture(span)
    if not isinstance(format_str, str):
        raise TypeError(f"debug.printf requires string format literal, but got {type(format_str).__name__}")

    normalized_args = [
        _normalize_expr(arg, actual_span, int_dtype=DataType.INDEX, float_dtype=DataType.FP32)
        if not isinstance(arg, Expr)
        else arg
        for arg in args
    ]

    specs = _scan_printf_format(format_str)
    if len(specs) != len(normalized_args):
        raise ValueError(
            f"printf format expects {len(specs)} scalar arguments, but got {len(normalized_args)}"
        )

    for idx, (spec, arg) in enumerate(zip(specs, normalized_args, strict=True)):
        scalar_type = arg.type
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"debug.printf argument {idx} requires ScalarType input, but got {type(scalar_type).__name__}"
            )

        conversion = spec[-1]
        if conversion in _INTEGER_CONVERSIONS and not _is_integer_dtype(scalar_type.dtype):
            raise TypeError(
                f"debug.printf conversion '{spec}' requires integer/index scalar, but got {scalar_type.dtype}"
            )
        if conversion in _FLOAT_CONVERSIONS and scalar_type.dtype != DataType.FP32:
            raise TypeError(
                f"debug.printf conversion '{spec}' requires FP32 scalar, but got {scalar_type.dtype}"
            )

    kwargs: dict[str, Any] = {"format": format_str}
    return _ir_core.create_op_call("debug.printf", normalized_args, kwargs, actual_span)


__all__ = ["printf_"]
