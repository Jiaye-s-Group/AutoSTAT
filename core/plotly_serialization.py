"""JSON-safe Plotly transport helpers.

Plotly accepts pandas ``Interval`` values in categorical traces, but neither
its standard JSON encoder nor ``orjson`` can serialize them.  Converting only
those transport values to their canonical interval label keeps the chart's
categories and bin boundaries intact while allowing figures to be cached,
rendered, and included in reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio


def _json_safe_plotly_value(value: Any) -> Any:
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, pd.IntervalIndex):
        return [str(item) for item in value]
    if isinstance(value, pd.Categorical):
        return [_json_safe_plotly_value(item) for item in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe_plotly_value(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return _json_safe_plotly_value(value.tolist())
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe_plotly_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_plotly_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_json_safe_plotly_value(item) for item in value)
    return value


def json_safe_figure(fig: go.Figure | dict[str, Any]) -> go.Figure:
    """Return a copy whose transport values are accepted by Plotly JSON."""
    raw_figure = go.Figure(fig)
    return go.Figure(_json_safe_plotly_value(raw_figure.to_plotly_json()))


def figure_to_json(fig: go.Figure | dict[str, Any]) -> str:
    """Serialize a figure after converting pandas Interval transport values."""
    return pio.to_json(json_safe_figure(fig))
