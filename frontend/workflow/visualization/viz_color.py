import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

PALETTES = {
    "Classic": [
        "#2B5C8A", "#4F81AF", "#77ACD3", "#D9D5C9", "#F69035"
    ],
    "Deep Sea": [
        "#e05b3d", "#c8e5ed", "#344e99", "#f5b46f", "#73a1c3"
    ],
    "Nature Editorial": [
        "#2F4858", "#33658A", "#86B3A3", "#F2C57C", "#D96C5F"
    ],
    "Ocean Breeze": [
        "#03045E", "#0077B6", "#00B4D8", "#90E0EF", "#CAF0F8"
    ],
    "Olive Garden Feast": [
        "#606C38", "#283618", "#FEFAE0", "#DDA15E", "#BC6C25"
    ],
    "Fiery Ocean": [
        "#780000", "#C1121F", "#FDF0D5", "#003049", "#669BBC"
    ],
    "Refreshing Summer Fun": [
        "#8ECAE6", "#219EBC", "#023047", "#FFB703", "#FB8500"
    ],
    "Golden Summer Fields": [
        "#5E4B3C", "#A67C52", "#D9C3A5", "#EADFCF", "#C46A4A"
    ],
    "Bold Berry": [
        "#F9DBBD", "#FFA5AB", "#DA627D", "#A53860", "#450920"
    ],
    "Fresh Greens": [
        "#D8F3DC", "#95D5B2", "#52B788", "#2D6A4F", "#1B4332"
    ],
}

CUSTOM_PALETTE_NAME = "自定义"
CUSTOM_PALETTE_SIZE = 5
CUSTOM_SWATCH_COMPONENT = components.declare_component(
    "viz_color_swatch",
    path=str(Path(__file__).resolve().parent / "viz_color_swatch_component"),
)
HISTOGRAM_BOUNDARY_SHAPE_NAME = "autostat_histogram_boundary"
BOX_MEDIAN_SHAPE_NAME = "autostat_box_median"
HEATMAP_NEUTRAL = "#F7F4EF"


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return hex_color

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return None
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _rgb_to_hex(rgb_color):
    if rgb_color is None or len(rgb_color) != 3:
        return None
    bounded = [max(0, min(255, int(round(channel)))) for channel in rgb_color]
    return "#{:02X}{:02X}{:02X}".format(*bounded)


def _darken_hex(hex_color, factor):
    rgb = _hex_to_rgb(hex_color)
    if rgb is None:
        return hex_color

    factor = max(0.0, min(1.0, factor))
    return _rgb_to_hex(tuple(channel * (1 - factor) for channel in rgb))


def _normalize_hex_color(hex_color, fallback="#FFFFFF"):
    if not isinstance(hex_color, str):
        return fallback

    normalized = hex_color.strip().upper()
    if len(normalized) != 7 or not normalized.startswith("#"):
        return fallback

    try:
        int(normalized[1:], 16)
    except ValueError:
        return fallback

    return normalized


def _relative_luminance(hex_color):
    rgb = _hex_to_rgb(hex_color)
    if rgb is None:
        return -1

    def convert(channel):
        channel = channel / 255
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (convert(channel) for channel in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _rotated_palette(colors, fig_index):
    if not colors:
        return []
    start = fig_index % len(colors)
    return list(colors[start:] + colors[:start])


def _trace_color(rotated_palette, trace_index):
    return rotated_palette[trace_index % len(rotated_palette)]


def _histogram_fill(base_color, fig_index, trace_index=0):
    alpha_steps = [0.6, 0.65, 0.7, 0.75, 0.8]
    alpha = alpha_steps[(fig_index + trace_index) % len(alpha_steps)]
    return _hex_to_rgba(base_color, alpha)


def _continuous_colorscale(colors):
    if not colors:
        return None
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    steps = len(colors) - 1
    return [[index / steps, color] for index, color in enumerate(colors)]


def _ordered_scale_colors(colors):
    unique_colors = []
    seen = set()
    for color in colors:
        if color in seen:
            continue
        unique_colors.append(color)
        seen.add(color)
    return sorted(unique_colors, key=_relative_luminance, reverse=True)


def _heatmap_colorscale(colors):
    ordered_colors = _ordered_scale_colors(colors)
    if not ordered_colors:
        return None
    if len(ordered_colors) == 1:
        return [[0.0, ordered_colors[0]], [1.0, ordered_colors[0]]]

    enriched_colors = [ordered_colors[0], *ordered_colors[1:-1], ordered_colors[-1]]
    if len(enriched_colors) >= 3:
        middle_index = len(enriched_colors) // 2
        enriched_colors[middle_index] = HEATMAP_NEUTRAL

    return _continuous_colorscale(enriched_colors)


def _default_custom_palette():
    return ["#FFFFFF"] * CUSTOM_PALETTE_SIZE


def _palette_name_from_colors(colors):
    if not isinstance(colors, (list, tuple)) or len(colors) != CUSTOM_PALETTE_SIZE:
        return list(PALETTES.keys())[0]

    normalized = [_normalize_hex_color(color) for color in colors]
    for name, palette in PALETTES.items():
        if normalized == [_normalize_hex_color(color) for color in palette]:
            return name
    return CUSTOM_PALETTE_NAME


def _initialize_palette_state(agent):
    stored_colors = agent.load_color()
    has_stored_palette = (
        isinstance(stored_colors, (list, tuple))
        and len(stored_colors) == CUSTOM_PALETTE_SIZE
    )
    default_palette = [_normalize_hex_color(color) for color in PALETTES[list(PALETTES.keys())[0]]]
    if has_stored_palette:
        normalized_stored_colors = [_normalize_hex_color(color) for color in stored_colors]
    else:
        normalized_stored_colors = default_palette

    if "viz_palette_choice" not in st.session_state or not has_stored_palette:
        st.session_state.viz_palette_choice = _palette_name_from_colors(normalized_stored_colors)

    if "viz_custom_palette" not in st.session_state or not has_stored_palette:
        if _palette_name_from_colors(normalized_stored_colors) == CUSTOM_PALETTE_NAME:
            st.session_state.viz_custom_palette = normalized_stored_colors
        else:
            st.session_state.viz_custom_palette = _default_custom_palette()


def _render_palette_preview(colors):
    cols = st.columns(len(colors))
    for col, code in zip(cols, colors):
        col.markdown(
            f"""
            <div style="
                background-color: {code};
                height: 30px;
                border-radius: 4px;
                margin-bottom: 2px;
            "></div>
            <div style="text-align: center; font-size: 10px;">{code}</div>
            """,
            unsafe_allow_html=True,
        )


def _render_custom_palette_picker():
    current_colors = [
        _normalize_hex_color(color)
        for color in st.session_state.get("viz_custom_palette", _default_custom_palette())
    ]
    colors = []
    cols = st.columns(CUSTOM_PALETTE_SIZE)
    for index, col in enumerate(cols):
        with col:
            component_value = CUSTOM_SWATCH_COMPONENT(
                color=current_colors[index],
                key=f"viz_custom_swatch_{index}",
                default=current_colors[index],
            )
            if isinstance(component_value, str):
                colors.append(_normalize_hex_color(component_value))
            else:
                colors.append(current_colors[index])

    st.session_state.viz_custom_palette = colors
    return colors


def _color_value_sequence(color_value):
    if color_value is None or isinstance(color_value, (str, bytes, dict)):
        return None

    if isinstance(color_value, np.ndarray):
        values = color_value.tolist()
    elif hasattr(color_value, "tolist") and not isinstance(color_value, (list, tuple)):
        try:
            values = color_value.tolist()
        except Exception:
            values = None
    else:
        try:
            values = list(color_value)
        except TypeError:
            values = None

    if not isinstance(values, list) or len(values) <= 1:
        return None

    return values


def _is_missing_color_value(value):
    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _hashable_color_key(value):
    if isinstance(value, np.generic):
        value = value.item()

    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _ordered_unique_values(values):
    unique_values = []
    seen = set()

    for value in values:
        if _is_missing_color_value(value):
            continue

        key = _hashable_color_key(value)
        if key in seen:
            continue

        seen.add(key)
        unique_values.append(value)

    return unique_values


def _is_integer_like(value):
    if isinstance(value, bool):
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return np.isfinite(numeric_value) and numeric_value.is_integer()


def _numeric_color_sequence(values):
    numeric_values = []
    has_multiple = False
    previous_value = None

    for value in values:
        if _is_missing_color_value(value):
            numeric_values.append(None)
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(numeric_value):
            return None

        numeric_values.append(numeric_value)
        if previous_value is not None and not np.isclose(previous_value, numeric_value):
            has_multiple = True
        previous_value = numeric_value

    return numeric_values if has_multiple else None


def _discrete_marker_colors(color_value, palette):
    values = _color_value_sequence(color_value)
    if not values:
        return None

    unique_values = _ordered_unique_values(values)
    if len(unique_values) <= 1:
        return None

    max_discrete_values = max(len(palette) * 2, 12)
    if len(unique_values) > max_discrete_values:
        return None

    if all(_is_integer_like(value) for value in unique_values):
        pass
    else:
        numeric_values = _numeric_color_sequence(unique_values)
        if numeric_values is not None:
            return None

    color_map = {
        _hashable_color_key(value): palette[index % len(palette)]
        for index, value in enumerate(unique_values)
    }
    return [
        None if _is_missing_color_value(value) else color_map[_hashable_color_key(value)]
        for value in values
    ]


def _split_trace_by_discrete_color(trace, palette, secondary_color):
    """Split a trace with discrete color encoding into per-category sub-traces.

    When marker.color is a categorical array, Plotly won't create per-category
    legend entries on its own.  This function returns a list of sub-traces,
    one per unique category value, each with:
      - its own ``name`` (= category label) so the legend gets a dedicated swatch
      - a single solid ``marker.color`` so the swatch matches the data points
      - ``legendgroup`` set to avoid duplicate entries if the same category
        appears in another sub-figure

    Returns ``None`` if the trace does not use discrete color encoding.
    """
    marker = getattr(trace, "marker", None)
    if marker is None:
        return None

    original_color_value = getattr(marker, "color", None)
    values = _color_value_sequence(original_color_value)
    if not values:
        return None

    unique_values = _ordered_unique_values(values)
    if len(unique_values) <= 1:
        return None

    max_discrete_values = max(len(palette) * 2, 12)
    if len(unique_values) > max_discrete_values:
        return None

    # Reject continuous numeric sequences
    if all(_is_integer_like(value) for value in unique_values):
        pass
    else:
        numeric_vals = _numeric_color_sequence(unique_values)
        if numeric_vals is not None:
            return None

    color_map = {
        _hashable_color_key(value): palette[index % len(palette)]
        for index, value in enumerate(unique_values)
    }

    trace_type = getattr(trace, "type", "")
    sub_traces = []
    for cat_value in unique_values:
        cat_key = _hashable_color_key(cat_value)
        cat_color = color_map[cat_key]
        mask = [
            (not _is_missing_color_value(v)) and _hashable_color_key(v) == cat_key
            for v in values
        ]

        # Clone the trace as a dict so we can slice data arrays
        trace_dict = trace.to_plotly_json()
        cat_label = str(cat_value)

        # Slice array-like data attributes
        for attr in ("x", "y", "text", "hovertext", "customdata", "ids"):
            arr = trace_dict.get(attr)
            if arr is not None and isinstance(arr, (list, tuple)) and len(arr) == len(mask):
                trace_dict[attr] = [arr[i] for i, m in enumerate(mask) if m]

        # Set single-color marker
        trace_dict.setdefault("marker", {})
        trace_dict["marker"]["color"] = cat_color
        trace_dict["marker"]["line"] = {"color": secondary_color, "width": 0.8}
        # Remove coloraxis / colorscale that no longer apply
        trace_dict["marker"].pop("coloraxis", None)
        trace_dict["marker"].pop("colorscale", None)
        trace_dict["marker"].pop("showscale", None)

        trace_dict["name"] = cat_label
        trace_dict["legendgroup"] = f"autostat_cat_{cat_label}"
        trace_dict["showlegend"] = True

        # Reconstruct as a proper go trace
        try:
            sub_trace = go.Figure(data=[trace_dict]).data[0]
        except Exception:
            return None
        sub_traces.append(sub_trace)

    return sub_traces


def _apply_existing_color_encoding(trace, palette, secondary_color):
    marker = getattr(trace, "marker", None)
    if marker is None:
        return False

    discrete_colors = _discrete_marker_colors(getattr(marker, "color", None), palette)
    if discrete_colors is not None:
        marker.color = discrete_colors
        if hasattr(marker, "coloraxis"):
            marker.coloraxis = None
        if hasattr(marker, "showscale"):
            marker.showscale = False
        if hasattr(marker, "line") and marker.line is not None:
            marker.line.color = secondary_color
            marker.line.width = 0.8
        return True

    numeric_values = _numeric_color_sequence(_color_value_sequence(getattr(marker, "color", None)) or [])
    if numeric_values is None:
        return False

    colorscale = _continuous_colorscale(palette)
    if colorscale is None:
        return False

    marker.color = numeric_values
    if hasattr(marker, "coloraxis"):
        marker.coloraxis = None
    if hasattr(marker, "colorscale"):
        marker.colorscale = colorscale
    if hasattr(marker, "showscale"):
        marker.showscale = True
    if hasattr(marker, "line") and marker.line is not None:
        marker.line.color = secondary_color
        marker.line.width = 0.8
    return True


def _axis_layout_name(axis_ref):
    if not axis_ref:
        return None
    if len(axis_ref) == 1:
        return f"{axis_ref}axis"
    return f"{axis_ref[0]}axis{axis_ref[1:]}"


def _coerce_numeric_values(values):
    if values is None:
        return None

    numeric_values = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            return None

    return numeric_values or None


def _resolve_histogram_edges(trace, values):
    numeric_values = np.asarray(values, dtype=float)
    if numeric_values.size == 0:
        return None

    orientation = getattr(trace, "orientation", "v")
    bin_config = getattr(trace, "ybins", None) if orientation == "h" else getattr(trace, "xbins", None)
    start = getattr(bin_config, "start", None) if bin_config is not None else None
    end = getattr(bin_config, "end", None) if bin_config is not None else None
    size = getattr(bin_config, "size", None) if bin_config is not None else None

    try:
        start = float(start) if start is not None else None
        end = float(end) if end is not None else None
        size = float(size) if size is not None else None
    except (TypeError, ValueError):
        start = end = size = None

    if start is not None and end is not None and size and size > 0:
        edges = np.arange(start, end + size, size, dtype=float)
        if edges.size >= 2:
            if edges[-1] < end:
                edges = np.append(edges, end)
            return edges

    if numeric_values.size == 1 or np.allclose(numeric_values.min(), numeric_values.max()):
        value = float(numeric_values[0])
        padding = max(abs(value) * 0.01, 0.5)
        return np.array([value - padding, value + padding], dtype=float)

    nbins = getattr(trace, "nbinsy", None) if orientation == "h" else getattr(trace, "nbinsx", None)
    bins = int(nbins) if isinstance(nbins, int) and nbins > 1 else "auto"
    edges = np.histogram_bin_edges(numeric_values, bins=bins)
    return edges if edges.size >= 2 else None


def _histogram_group_key(trace):
    orientation = getattr(trace, "orientation", "v")
    axis_ref = getattr(trace, "yaxis", "y") if orientation == "h" else getattr(trace, "xaxis", "x")
    nbins = getattr(trace, "nbinsy", None) if orientation == "h" else getattr(trace, "nbinsx", None)
    bin_config = getattr(trace, "ybins", None) if orientation == "h" else getattr(trace, "xbins", None)

    return (
        orientation,
        axis_ref,
        getattr(trace, "bingroup", None),
        nbins,
        getattr(bin_config, "start", None) if bin_config is not None else None,
        getattr(bin_config, "end", None) if bin_config is not None else None,
        getattr(bin_config, "size", None) if bin_config is not None else None,
    )


def _build_histogram_edge_map(fig):
    grouped_values = {}
    grouped_trace = {}

    for trace in fig.data:
        if getattr(trace, "type", "") != "histogram":
            continue

        histfunc = getattr(trace, "histfunc", None)
        histnorm = getattr(trace, "histnorm", None)
        if histfunc not in (None, "", "count") or histnorm not in (None, ""):
            continue

        orientation = getattr(trace, "orientation", "v")
        axis_values = trace.y if orientation == "h" else trace.x
        numeric_values = _coerce_numeric_values(axis_values)
        if not numeric_values:
            continue

        key = _histogram_group_key(trace)
        grouped_values.setdefault(key, []).extend(numeric_values)
        grouped_trace.setdefault(key, trace)

    edge_map = {}
    for key, values in grouped_values.items():
        edge_map[key] = _resolve_histogram_edges(grouped_trace[key], values)

    return edge_map


def _convert_histogram_to_bar(trace, fill_color, shared_edges=None):
    histfunc = getattr(trace, "histfunc", None)
    histnorm = getattr(trace, "histnorm", None)
    if histfunc not in (None, "", "count") or histnorm not in (None, ""):
        return None, None, None

    orientation = getattr(trace, "orientation", "v")
    axis_values = trace.y if orientation == "h" else trace.x
    numeric_values = _coerce_numeric_values(axis_values)
    if not numeric_values:
        return None, None, None

    edges = shared_edges if shared_edges is not None else _resolve_histogram_edges(trace, numeric_values)
    if edges is None:
        return None, None, None

    counts, edges = np.histogram(np.asarray(numeric_values, dtype=float), bins=edges)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    widths = np.diff(edges).tolist()

    common_kwargs = {
        "name": getattr(trace, "name", None),
        "showlegend": getattr(trace, "showlegend", None),
        "legendgroup": getattr(trace, "legendgroup", None),
        "offsetgroup": getattr(trace, "offsetgroup", None),
        "alignmentgroup": getattr(trace, "alignmentgroup", None),
        "meta": {"autostat_histogram": True},
        "opacity": getattr(trace, "opacity", None),
        "hovertemplate": getattr(trace, "hovertemplate", None),
        "xaxis": getattr(trace, "xaxis", None),
        "yaxis": getattr(trace, "yaxis", None),
        "marker": {
            "color": fill_color,
            "line": {"color": "#000000", "width": 0.5},
        },
    }

    # Plotly legend swatches render marker.color literally; when fill_color
    # is an rgba with alpha < 1, the legend swatch looks faded / mismatched
    # against the actual bars (which are composited on a white/dark bg).
    # Extracting the base hex and setting it on the trace-level `legend*`
    # attributes isn't supported by Plotly, so we instead move the
    # semi-transparency from marker.color to trace-level opacity.  This way
    # the legend swatch uses the opaque base color and matches the bars.
    _rgba_match = None
    if isinstance(fill_color, str) and fill_color.startswith("rgba("):
        import re as _re
        _rgba_match = _re.match(
            r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)",
            fill_color,
        )
    if _rgba_match:
        _r, _g, _b = int(_rgba_match.group(1)), int(_rgba_match.group(2)), int(_rgba_match.group(3))
        _a = float(_rgba_match.group(4))
        _opaque_hex = "#{:02X}{:02X}{:02X}".format(_r, _g, _b)
        common_kwargs["marker"]["color"] = _opaque_hex
        # Combine with any existing trace-level opacity
        existing_opacity = common_kwargs.get("opacity")
        if existing_opacity is not None:
            try:
                common_kwargs["opacity"] = float(existing_opacity) * _a
            except (TypeError, ValueError):
                common_kwargs["opacity"] = _a
        else:
            common_kwargs["opacity"] = _a

    if orientation == "h":
        bar_trace = go.Bar(
            x=counts.tolist(),
            y=centers,
            width=widths,
            orientation="h",
            **common_kwargs,
        )
        axis_ref = getattr(trace, "yaxis", "y")
    else:
        bar_trace = go.Bar(
            x=centers,
            y=counts.tolist(),
            width=widths,
            **common_kwargs,
        )
        axis_ref = getattr(trace, "xaxis", "x")

    return bar_trace, axis_ref, edges


def _pad_histogram_axis(fig, axis_ref, edges):
    axis_name = _axis_layout_name(axis_ref)
    if not axis_name:
        return

    axis = getattr(fig.layout, axis_name, None)
    if axis is not None and axis.range is not None:
        return

    if edges is None or len(edges) < 2:
        return

    widths = np.diff(np.asarray(edges, dtype=float))
    min_width = float(widths.min()) if widths.size else 1.0
    padding = max(min_width * 0.01, 1e-9)
    getattr(fig.layout, axis_name).range = [float(edges[0] - padding), float(edges[-1] + padding)]


def _histogram_boundary_shapes(trace):
    trace_meta = getattr(trace, "meta", None)
    if not (getattr(trace, "type", "") == "bar" and isinstance(trace_meta, dict) and trace_meta.get("autostat_histogram")):
        return []

    orientation = getattr(trace, "orientation", "v")
    x_data = getattr(trace, "x", None)
    y_data = getattr(trace, "y", None)
    width_data = getattr(trace, "width", None)
    x_values = list(x_data) if x_data is not None else []
    y_values = list(y_data) if y_data is not None else []
    widths = list(width_data) if width_data is not None else []
    if not widths:
        return []

    min_width = min(float(width) for width in widths if width is not None)
    inset = max(min_width * 0.002, 1e-9)
    line_style = {"color": "#000000", "width": 0.5}

    if orientation == "h":
        if not y_values or not x_values:
            return []
        lower_edge = float(y_values[0] - widths[0] / 2 + inset)
        upper_edge = float(y_values[-1] + widths[-1] / 2 - inset)
        return [
            go.layout.Shape(
                type="line",
                name=HISTOGRAM_BOUNDARY_SHAPE_NAME,
                xref=getattr(trace, "xaxis", "x"),
                yref=getattr(trace, "yaxis", "y"),
                x0=0,
                x1=float(x_values[0]),
                y0=lower_edge,
                y1=lower_edge,
                line=line_style,
            ),
            go.layout.Shape(
                type="line",
                name=HISTOGRAM_BOUNDARY_SHAPE_NAME,
                xref=getattr(trace, "xaxis", "x"),
                yref=getattr(trace, "yaxis", "y"),
                x0=0,
                x1=float(x_values[-1]),
                y0=upper_edge,
                y1=upper_edge,
                line=line_style,
            ),
        ]

    if not x_values or not y_values:
        return []

    left_edge = float(x_values[0] - widths[0] / 2 + inset)
    right_edge = float(x_values[-1] + widths[-1] / 2 - inset)
    return [
        go.layout.Shape(
            type="line",
            name=HISTOGRAM_BOUNDARY_SHAPE_NAME,
            xref=getattr(trace, "xaxis", "x"),
            yref=getattr(trace, "yaxis", "y"),
            x0=left_edge,
            x1=left_edge,
            y0=0,
            y1=float(y_values[0]),
            line=line_style,
        ),
        go.layout.Shape(
            type="line",
            name=HISTOGRAM_BOUNDARY_SHAPE_NAME,
            xref=getattr(trace, "xaxis", "x"),
            yref=getattr(trace, "yaxis", "y"),
            x0=right_edge,
            x1=right_edge,
            y0=0,
            y1=float(y_values[-1]),
            line=line_style,
        ),
    ]


def _trace_axis_ref(trace, axis_name):
    return getattr(trace, axis_name, axis_name[0])


def _to_list(values):
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values.tolist()
    if hasattr(values, "tolist") and not isinstance(values, list):
        try:
            converted = values.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass
    try:
        return list(values)
    except TypeError:
        return None


def _default_box_anchor(trace, orientation):
    if orientation == "h":
        anchor = getattr(trace, "y0", None)
    else:
        anchor = getattr(trace, "x0", None)

    if anchor not in (None, ""):
        return anchor

    name = getattr(trace, "name", None)
    if name not in (None, ""):
        return name

    return " "


def _box_group_medians(trace):
    orientation = getattr(trace, "orientation", "v")
    distribution_values = _to_list(trace.x if orientation == "h" else trace.y)
    position_values = _to_list(trace.y if orientation == "h" else trace.x)

    medians = _to_list(getattr(trace, "median", None))
    if medians:
        if position_values and len(position_values) == len(medians):
            return list(zip(position_values, medians))

        if len(medians) == 1:
            return [(_default_box_anchor(trace, orientation), medians[0])]

        step = getattr(trace, "dy" if orientation == "h" else "dx", None)
        start = getattr(trace, "y0" if orientation == "h" else "x0", None)
        if start is not None and isinstance(step, (int, float)):
            return [(start + step * index, value) for index, value in enumerate(medians)]

        return []

    if not distribution_values:
        return []

    if position_values and len(position_values) == len(distribution_values):
        grouped = {}
        order = []
        for anchor, value in zip(position_values, distribution_values):
            if _is_missing_color_value(value):
                continue
            key = _hashable_color_key(anchor)
            if key not in grouped:
                grouped[key] = {"anchor": anchor, "values": []}
                order.append(key)
            grouped[key]["values"].append(float(value))

        specs = []
        for key in order:
            values = grouped[key]["values"]
            if values:
                specs.append((grouped[key]["anchor"], float(np.median(values))))
        return specs

    cleaned_values = []
    for value in distribution_values:
        if _is_missing_color_value(value):
            continue
        try:
            cleaned_values.append(float(value))
        except (TypeError, ValueError):
            return []

    if not cleaned_values:
        return []

    return [(_default_box_anchor(trace, orientation), float(np.median(cleaned_values)))]


def _box_distribution_values(trace):
    orientation = getattr(trace, "orientation", "v")
    distribution_values = _to_list(trace.x if orientation == "h" else trace.y)
    if not distribution_values:
        return []

    cleaned_values = []
    for value in distribution_values:
        if _is_missing_color_value(value):
            continue
        try:
            cleaned_values.append(float(value))
        except (TypeError, ValueError):
            return []

    return cleaned_values


def _is_collapsed_box_trace(trace):
    values = _box_distribution_values(trace)
    if not values:
        return False

    first_value = values[0]
    return all(np.isclose(first_value, value) for value in values[1:])


def _box_median_overlay_trace(trace, median_color):
    if getattr(trace, "type", "") != "box":
        return None

    specs = _box_group_medians(trace)
    if not specs:
        return None

    anchors = [anchor for anchor, _ in specs]
    medians = [median for _, median in specs]
    orientation = getattr(trace, "orientation", "v")

    overlay_kwargs = {
        "q1": medians,
        "median": medians,
        "q3": medians,
        "lowerfence": medians,
        "upperfence": medians,
        "orientation": orientation,
        "width": getattr(trace, "width", None),
        "offsetgroup": getattr(trace, "offsetgroup", None),
        "alignmentgroup": getattr(trace, "alignmentgroup", None),
        "xaxis": getattr(trace, "xaxis", None),
        "yaxis": getattr(trace, "yaxis", None),
        "hoverinfo": "skip",
        "showlegend": False,
        "boxpoints": False,
        "whiskerwidth": 0,
        "fillcolor": "rgba(0, 0, 0, 0)",
        "line": {"color": median_color, "width": 1.9},
        "marker": {"opacity": 0},
        "name": getattr(trace, "name", None),
        "meta": {"autostat_box_median_overlay": True},
    }

    if orientation == "h":
        if getattr(trace, "y", None) is None:
            overlay_kwargs["y0"] = getattr(trace, "y0", None)
        else:
            overlay_kwargs["y"] = anchors
        overlay_kwargs["x0"] = getattr(trace, "x0", None)
    else:
        if getattr(trace, "x", None) is None:
            overlay_kwargs["x0"] = getattr(trace, "x0", None)
        else:
            overlay_kwargs["x"] = anchors
        overlay_kwargs["y0"] = getattr(trace, "y0", None)

    return go.Box(**overlay_kwargs)


def apply_palette_to_figure(fig, colors, fig_index=0):
    if isinstance(fig, str):
        try:
            fig = pio.from_json(fig)
        except Exception:
            return fig
    elif isinstance(fig, dict):
        try:
            fig = go.Figure(fig)
        except Exception:
            return fig
    elif not isinstance(fig, go.Figure):
        return fig

    if not colors:
        return go.Figure(fig)

    updated_fig = go.Figure(fig)
    palette = _rotated_palette(list(colors), fig_index)
    new_traces = []
    median_overlay_traces = []
    converted_histogram = False
    histogram_edge_map = _build_histogram_edge_map(updated_fig)

    for trace_index, trace in enumerate(updated_fig.data):
        primary_color = _trace_color(palette, trace_index)
        secondary_color = _trace_color(palette, trace_index + 1)
        tertiary_color = _trace_color(palette, trace_index + 2)
        trace_mode = getattr(trace, "mode", "") or ""
        trace_type = getattr(trace, "type", "")
        is_box_median_overlay = False
        _color_encoding_applied = False
        point_count = 0
        for attr in ("x", "y", "labels", "values"):
            values = getattr(trace, attr, None)
            if values is None:
                continue
            try:
                point_count = len(values)
                break
            except TypeError:
                continue

        if (
            hasattr(trace, "marker")
            and trace.marker is not None
            and trace_type not in {"heatmap", "histogram2d", "histogram2dcontour", "contour", "heatmapgl"}
        ):
            trace_meta = getattr(trace, "meta", None)
            is_histogram_bar = isinstance(trace_meta, dict) and trace_meta.get("autostat_histogram")
            is_box_median_overlay = isinstance(trace_meta, dict) and trace_meta.get("autostat_box_median_overlay")
            if trace_type in {"pie", "sunburst", "treemap", "funnelarea"}:
                color_count = point_count or 1
                trace.marker.colors = [palette[(trace_index + i) % len(palette)] for i in range(color_count)]
                trace.marker.line.color = secondary_color
                trace.marker.line.width = 1
            elif trace_type == "histogram":
                histogram_fill = _histogram_fill(primary_color, fig_index, trace_index)
                shared_edges = histogram_edge_map.get(_histogram_group_key(trace))
                bar_trace, axis_ref, edges = _convert_histogram_to_bar(trace, histogram_fill, shared_edges)
                if bar_trace is not None:
                    _pad_histogram_axis(updated_fig, axis_ref, edges)
                    new_traces.append(bar_trace)
                    converted_histogram = True
                    continue
                trace.marker.color = histogram_fill
                trace.marker.line.color = "#000000"
                trace.marker.line.width = 0.5
            elif trace_type == "bar" and is_histogram_bar:
                histogram_fill = _histogram_fill(primary_color, fig_index, trace_index)
                trace.marker.color = histogram_fill
                trace.marker.line.color = "#000000"
                trace.marker.line.width = 0.5
            elif trace_type == "bar":
                trace.marker.color = primary_color
                trace.marker.line.color = secondary_color
                trace.marker.line.width = 1
            elif trace_type == "splom" or (
                trace_type.startswith("scatter") and "markers" in trace_mode
            ):
                split_traces = _split_trace_by_discrete_color(trace, palette, secondary_color)
                if split_traces is not None:
                    new_traces.extend(split_traces)
                    _color_encoding_applied = True
                    continue
                elif _apply_existing_color_encoding(trace, palette, secondary_color):
                    _color_encoding_applied = True
                else:
                    trace.marker.color = primary_color
                    trace.marker.line.color = secondary_color
                    trace.marker.line.width = 0.8
            elif trace_type.startswith("scatter"):
                split_traces = _split_trace_by_discrete_color(trace, palette, secondary_color)
                if split_traces is not None:
                    new_traces.extend(split_traces)
                    _color_encoding_applied = True
                    continue
                elif _apply_existing_color_encoding(trace, palette, secondary_color):
                    _color_encoding_applied = True
                else:
                    trace.marker.color = primary_color
                    trace.marker.line.color = secondary_color
                    trace.marker.line.width = 0.8
            elif _apply_existing_color_encoding(trace, palette, secondary_color):
                _color_encoding_applied = True
            else:
                trace.marker.color = primary_color
                if hasattr(trace.marker, "line") and trace.marker.line is not None:
                    trace.marker.line.color = secondary_color
                    trace.marker.line.width = 1

        if trace_type in {"heatmap", "histogram2d", "histogram2dcontour", "contour", "heatmapgl"}:
            colorscale = _heatmap_colorscale(palette)
            if colorscale is not None:
                if hasattr(trace, "colorscale"):
                    trace.colorscale = colorscale
                coloraxis_ref = getattr(trace, "coloraxis", None)
                if coloraxis_ref:
                    coloraxis_name = coloraxis_ref if isinstance(coloraxis_ref, str) else str(coloraxis_ref)
                    coloraxis = getattr(updated_fig.layout, coloraxis_name, None)
                    if coloraxis is not None:
                        coloraxis.colorscale = colorscale

        if trace_type == "histogram2dcontour" and hasattr(trace, "line") and trace.line is not None:
            trace.line.color = primary_color
            if hasattr(trace, "contours") and trace.contours is not None:
                try:
                    trace.contours.coloring = "lines"
                except Exception:
                    pass

        if hasattr(trace, "line") and trace.line is not None and not _color_encoding_applied:
            trace.line.color = primary_color

        if hasattr(trace, "fillcolor") and not _color_encoding_applied:
            trace.fillcolor = _hex_to_rgba(primary_color, 0.65)

        if trace_type == "box" and is_box_median_overlay:
            pass
        elif trace_type == "box":
            is_collapsed_box = _is_collapsed_box_trace(trace)
            border_color = _darken_hex(primary_color, 0.18)
            median_color = "#FFFFFF"
            point_fill = _hex_to_rgba(primary_color, 0.42)
            trace.line.color = border_color
            trace.line.width = 1.1
            trace.whiskerwidth = 0.45
            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.color = point_fill
                trace.marker.size = 6
                if hasattr(trace.marker, "opacity"):
                    trace.marker.opacity = 0.9
                if hasattr(trace.marker, "outliercolor"):
                    trace.marker.outliercolor = point_fill
                if hasattr(trace.marker, "line") and trace.marker.line is not None:
                    trace.marker.line.color = border_color
                    trace.marker.line.width = 0.8
                    if hasattr(trace.marker.line, "outliercolor"):
                        trace.marker.line.outliercolor = border_color
                    if hasattr(trace.marker.line, "outlierwidth"):
                        trace.marker.line.outlierwidth = 0.8
            if hasattr(trace, "fillcolor"):
                trace.fillcolor = _hex_to_rgba(primary_color, 0.6)
            if hasattr(trace, "notched"):
                trace.notched = False
            if hasattr(trace, "boxmean"):
                trace.boxmean = False
            if is_collapsed_box:
                trace.boxpoints = "all"
                trace.jitter = 0.3
                trace.pointpos = 0
                if hasattr(trace, "fillcolor"):
                    trace.fillcolor = "rgba(0, 0, 0, 0)"
                if hasattr(trace, "marker") and trace.marker is not None:
                    trace.marker.color = primary_color
                    trace.marker.size = 8
                    if hasattr(trace.marker, "opacity"):
                        trace.marker.opacity = 1
            else:
                overlay_trace = _box_median_overlay_trace(trace, median_color)
                if overlay_trace is not None:
                    median_overlay_traces.append(overlay_trace)
        elif trace_type == "violin":
            border_color = _darken_hex(primary_color, 0.12)
            trace.line.color = border_color
            trace.line.width = 0.9
            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.color = primary_color
                if hasattr(trace.marker, "line") and trace.marker.line is not None:
                    trace.marker.line.color = border_color
                    trace.marker.line.width = 0.7
            if hasattr(trace, "fillcolor"):
                trace.fillcolor = _hex_to_rgba(primary_color, 0.5)
            elif hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.color = primary_color
            if hasattr(trace, "meanline") and trace.meanline is not None:
                trace.meanline.visible = True
                trace.meanline.color = "#FFFFFF"
                trace.meanline.width = 2.0
            if hasattr(trace, "box") and trace.box is not None:
                trace.box.visible = True
                trace.box.width = 0.2
                trace.box.fillcolor = "rgba(255, 255, 255, 0)"
                trace.box.line.color = "rgba(255, 255, 255, 0.9)"
                trace.box.line.width = 1.0
        elif trace_type == "scatter" and "lines" in trace_mode:
            trace.line.color = primary_color

        new_traces.append(trace)

    rebuilt_fig = go.Figure(layout=updated_fig.layout)
    for trace in new_traces:
        rebuilt_fig.add_trace(trace)
    for trace in median_overlay_traces:
        rebuilt_fig.add_trace(trace)

    preserved_shapes = [
        shape
        for shape in (rebuilt_fig.layout.shapes or ())
        if getattr(shape, "name", None) not in {HISTOGRAM_BOUNDARY_SHAPE_NAME, BOX_MEDIAN_SHAPE_NAME}
    ]
    histogram_shapes = []
    for trace in rebuilt_fig.data:
        histogram_shapes.extend(_histogram_boundary_shapes(trace))
    rebuilt_fig.update_layout(shapes=tuple(preserved_shapes + histogram_shapes))

    if converted_histogram:
        rebuilt_fig.update_layout(bargap=0)
    if median_overlay_traces:
        rebuilt_fig.update_layout(boxmode="overlay")

    return rebuilt_fig

def vis_palette(agent):
    _initialize_palette_state(agent)
    palette_options = [*PALETTES.keys(), CUSTOM_PALETTE_NAME]
    choice = st.selectbox("请选择配色方案（内含自定义）", palette_options, key="viz_palette_choice")

    if choice == CUSTOM_PALETTE_NAME:
        colors = _render_custom_palette_picker()
    else:
        colors = [_normalize_hex_color(color) for color in PALETTES[choice]]
        _render_palette_preview(colors)

    agent.save_color(colors)

    fig_desc_list = agent.load_fig() or []
    if fig_desc_list:
        recolored_figs = []
        for fig_index, item in enumerate(fig_desc_list):
            base_fig = item.get("base_fig", item.get("fig"))
            recolored_figs.append(
                {
                    "fig": apply_palette_to_figure(base_fig, colors, fig_index),
                    "base_fig": base_fig,
                    "desc": item.get("desc"),
                }
            )
        agent.save_fig(recolored_figs)

    return colors
