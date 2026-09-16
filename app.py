"""MyMaps - visualize your land, sectors, zones and points of interest from a CSV file on a map.

The CSV must contain at least an "WKT" column with WKT geometry
(POINT / LINESTRING / POLYGON) plus optional "name" and "description" columns.
"""

import csv
import html
import io
import re
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from shapely import wkt
from shapely.geometry import mapping
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from planner import (
    TILE_LAYERS,
    compute_altitudes,
    fmt_alt,
    get_land_polygon,
    render_planner_tab,
)

SAMPLE_CSV = BASE_DIR / "sample.csv"

# ---------------------------------------------------------------------------
# Category definitions: colour / legend / style used for each map layer.
# ---------------------------------------------------------------------------
CATEGORY_CONFIG = {
    "Land":     {"color": "#d73027", "legend": "Land boundary",  "weight": 4.0, "fill_opacity": 0.15, "dash": None},
    "Sector":   {"color": "#f07d00", "legend": "Sector",         "weight": 2.5, "fill_opacity": 0.18, "dash": "5,5"},
    "Zone":     {"color": "#2b83ba", "legend": "Zone",           "weight": 2.0, "fill_opacity": 0.25, "dash": "2,4"},
    "Route":    {"color": "#984ea3", "legend": "Route / pipe",   "weight": 3.0, "fill_opacity": 0.00, "dash": None},
    "Water":    {"color": "#1f6feb", "legend": "Water point",    "weight": 2.5, "fill_opacity": 0.35, "dash": None},
    "Position": {"color": "#2ea043", "legend": "Position / structure", "weight": 2.0, "fill_opacity": 0.35, "dash": None},
}

LAYER_NAMES = {
    "Land": "Land boundary",
    "Sector": "Sectors",
    "Zone": "Zones",
    "Route": "Routes",
    "Water": "Water",
    "Position": "Positions",
}

# Keywords (lower case) used to auto-classify features.
LAND_KW = ("land", "boundary", "terrain", "perimetre", "périmètre", "parcelle")
SECTOR_KW = ("sector", "secteur")
ZONE_KW = ("zone",)
WATER_KW = ("weel", "well", "puit", "water", "source", "reservoir", "basin")
ROUTE_KW = ("route", "trachee", "tranchée", "tranchee", "pipe", "tuyau", "line", "canal")

# Keywords (lower case) used to auto-classify features.
LAND_KW = ("land", "boundary", "terrain", "perimetre", "périmètre", "parcelle")
SECTOR_KW = ("sector", "secteur")
ZONE_KW = ("zone",)
WATER_KW = ("weel", "well", "puit", "water", "source", "reservoir", "basin")
ROUTE_KW = ("route", "trachee", "tranchée", "tranchee", "pipe", "tuyau", "line", "canal")

_GEO_TYPES = ("MULTIPOLYGON", "MULTILINESTRING", "MULTIPOINT", "POLYGON", "LINESTRING", "POINT")


# ---------------------------------------------------------------------------
# CSV / WKT parsing
# ---------------------------------------------------------------------------
def extract_wkt(text: str):
    """Pull the WKT piece out of a raw csv line, even if quoting is broken.

    Returns (wkt_string, remaining_text) or (None, text) when no WKT is found.
    """
    text = text.strip().strip('"')
    m = re.search(r"(MULTIPOLYGON|MULTILINESTRING|MULTIPOINT|POLYGON|LINESTRING|POINT)\b", text)
    if not m:
        return None, text
    start = m.start()
    i = m.end()
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    return text[start:i].strip(), text[i:].strip().strip('"')


def parse_csv(source) -> pd.DataFrame:
    """Parse an uploaded CSV file into a DataFrame with parsed WKT geometry."""
    if source is None:
        return pd.DataFrame(columns=["wkt", "name", "description", "geometry"])
    raw = source.getvalue().decode("utf-8-sig")

    rows = []
    try:
        tokens_rows = list(csv.reader(io.StringIO(raw)))
    except Exception:
        tokens_rows = []

    # remove completely empty lines
    tokens_rows = [r for r in tokens_rows if any(c.strip() for c in r)]

    header = 0
    if tokens_rows and tokens_rows[0] and "wkt" in tokens_rows[0][0].strip().lower():
        header = 1

    for line in tokens_rows[header:]:
        if len(line) >= 3:
            wkt_str, name, desc = line[0].strip(), line[1].strip(), line[2].strip()
        else:
            # lenient re-parsing of a possibly unquoted line
            joined = ",".join(line)
            wkt_str, rest = extract_wkt(joined)
            parts = [p.strip() for p in rest.split(",", 1)] if rest else []
            name = parts[0] if parts else ""
            desc = parts[1] if len(parts) > 1 else ""
        if not wkt_str:
            continue
        try:
            geom = wkt.loads(wkt_str)
        except Exception:
            geom = None
        if geom is None:
            continue
        rows.append({"wkt": wkt_str, "name": name, "description": desc, "geometry": geom})

    if not rows:
        return pd.DataFrame(columns=["wkt", "name", "description", "geometry"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_feature(name: str, description: str, geom, is_largest_polygon: bool) -> str:
    text = f"{name} {description}".lower()
    if any(k in text for k in LAND_KW):
        return "Land"
    if any(k in text for k in SECTOR_KW):
        return "Sector"
    if any(k in text for k in ZONE_KW):
        return "Zone"
    if any(k in text for k in WATER_KW):
        return "Water"
    if any(k in text for k in ROUTE_KW) or geom.geom_type == "LineString":
        return "Route"
    if geom.geom_type == "Point":
        return "Position"
    if geom.geom_type == "Polygon":
        # biggest polygon without a role is assumed to be the land parcel
        return "Land" if is_largest_polygon else "Position"
    return "Position"


def build_features(df: pd.DataFrame):
    """Assign each feature to a category and split polygons/lines (GeoJSON)
    from points (markers)."""
    if df.empty:
        return {}, {}

    poly_areas = [
        g.area for g in df["geometry"]
        if getattr(g, "geom_type", "") in ("Polygon", "MultiPolygon")
    ]
    largest_area = max(poly_areas) if poly_areas else 0

    features = {cat: [] for cat in CATEGORY_CONFIG}
    markers = {cat: [] for cat in CATEGORY_CONFIG}

    for row in df.itertuples():
        g = row.geometry
        any_role = any(k in f"{row.name} {row.description}".lower()
                       for k in (*LAND_KW, *SECTOR_KW, *ZONE_KW, *WATER_KW, *ROUTE_KW))
        is_largest = (g.area == largest_area and g.area > 0) if g.geom_type == "Polygon" else False
        cat = classify_feature(row.name, row.description, g, is_largest)
        props = {"name": row.name, "description": row.description, "_idx": row.Index}
        if g.geom_type == "Point":
            markers[cat].append((g, props))
        else:
            features[cat].append((g, props))

    # if a sole polygon was implicitly made "Land" but the CSV already has a
    # clearly labelled land / boundary feature, demote it to a Position
    land_marked = any(k in f"{r.name} {r.description}".lower() for r in df.itertuples()
                      for k in LAND_KW)
    if land_marked:
        # keep only explicit land features in the Land layer
        for g, props in features["Land"]:
            if not any(k in f"{props['name']} {props['description']}".lower() for k in LAND_KW):
                features["Position"].append((g, props))
        features["Land"] = [
            p for p in features["Land"]
            if any(k in f"{p[1]['name']} {p[1]['description']}".lower() for k in LAND_KW)
        ]
    return features, markers


def categorize_df(df: pd.DataFrame, alts: dict | None = None) -> pd.DataFrame:
    """Prepared table: category + altitude added to each parsed feature."""
    alts = alts or {}
    if df.empty:
        return pd.DataFrame(columns=["name", "description", "altitude_m", "geometry"])
    view = df[["name", "description"]].copy()
    view.insert(0, "category", df.apply(
        lambda r: classify_feature(
            r["name"], r["description"], r["geometry"],
            r["geometry"].area == max(
                (g.area for g in df["geometry"]
                 if g.geom_type in ("Polygon", "MultiPolygon")), default=0
            ) and r["geometry"].area > 0,
        ), axis=1,
    ))
    view["altitude_m"] = df.index.map(lambda i: alts.get(i))
    view["geometry"] = df["geometry"].apply(lambda g: g.wkt)
    return view


def processed_csv_string(view: pd.DataFrame) -> str:
    """WKT,name,description first (input format) + category,altitude_m extras."""
    out = view.copy()
    out.insert(4, "WKT", out.pop("geometry"))
    return out[["WKT", "name", "description", "category", "altitude_m"]].to_csv(
        index=False, lineterminator="\n"
    )


def compute_bounds(df: pd.DataFrame):
    xs, ys = [], []
    for g in df["geometry"]:
        try:
            coords = list(g.coords)
        except NotImplementedError:
            coords = []
            if g.geom_type == "Polygon":
                coords = list(g.exterior.coords)
            elif g.geom_type == "MultiPolygon":
                for poly in g.geoms:
                    coords.extend(poly.exterior.coords)
        for c in coords:
            if len(c) < 2:
                continue
            xs.append(c[0])
            ys.append(c[1])
    if not xs:
        return None
    return [[min(ys), min(xs)], [max(ys), max(xs)]]


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------
def build_map(df: pd.DataFrame, tile_name: str, active: list, show_labels: bool,
              alts: dict | None = None):
    tile = TILE_LAYERS[tile_name]
    alts = alts or {}
    known_alts = any(v is not None for v in alts.values())
    bounds = compute_bounds(df)
    if bounds:
        center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    else:
        center, bounds = [33.617, -4.727], None

    m = folium.Map(location=center, zoom_start=18, tiles=None, control_scale=True)

    for name, cfg in TILE_LAYERS.items():
        folium.TileLayer(
            tiles=cfg["url"],
            attr=cfg["attr"],
            name=name,
            show=(name == tile_name),
        ).add_to(m)
    folium.FitBounds(bounds) if bounds else None

    features, markers = build_features(df)

    for cat in active:
        cfg = CATEGORY_CONFIG[cat]
        if features.get(cat):
            fc = {
                "type": "FeatureCollection",
                "features": [],
            }
            for g, p in features[cat]:
                props = {"name": p["name"], "description": p["description"]}
                if known_alts:
                    props["altitude"] = fmt_alt(alts.get(p["_idx"]))
                fc["features"].append(
                    {"type": "Feature", "geometry": mapping(g), "properties": props}
                )
            style = lambda f, _cfg=cfg: {
                "color": _cfg["color"],
                "weight": _cfg["weight"],
                "fillColor": _cfg["color"],
                "fillOpacity": 0.0 if f["geometry"]["type"] == "LineString" else _cfg["fill_opacity"],
                "dashArray": _cfg["dash"] or "",
            }
            tooltip_fields = ["name", "description"]
            tooltip_aliases = ["Name", "Description"]
            if known_alts:
                tooltip_fields.append("altitude")
                tooltip_aliases.append("Altitude")
            geo = folium.GeoJson(
                fc,
                name=LAYER_NAMES[cat],
                style_function=style,
                highlight_function=lambda f: {"weight": 6, "color": "#333333"},
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    sticky=True,
                ),
            )
            geo.add_to(m)
            if show_labels:
                for g, p in features[cat]:
                    c = g.centroid
                    label = html.escape(f"{p['name']}: {p['description']}")
                    folium.Marker(
                        location=[c.y, c.x],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size:10px;color:#000;text-shadow:0 0 3px #fff;'
                                 f'white-space:nowrap;">{label}</div>'
                        ),
                    ).add_to(m)

        if markers.get(cat):
            mg = folium.FeatureGroup(name=LAYER_NAMES[cat])
            for g, p in markers[cat]:
                alt_txt = fmt_alt(alts.get(p["_idx"])) if known_alts else ""
                tip = f"{p['name']} - {p['description']}"
                popup_html = f"<b>{html.escape(p['name'])}</b><br>{html.escape(p['description'])}"
                if alt_txt:
                    tip += f" | Alt: {alt_txt}"
                    popup_html += f"<br><b>Altitude:</b> {html.escape(alt_txt)}"
                folium.CircleMarker(
                    location=[g.y, g.x],
                    radius=7,
                    color=cfg["color"],
                    weight=2,
                    fill=True,
                    fill_color=cfg["color"],
                    fill_opacity=0.9,
                    tooltip=tip,
                    popup=popup_html,
                ).add_to(mg)
            mg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MyMaps", layout="wide")

st.title("MyMaps")
st.caption("Load a CSV (WKT, name, description) and see it on the map.")

col_file, col_tile, col_filters, col_legend = st.columns([2, 1.4, 1.6, 1.2])

with col_file:
    data_src = st.radio(
        "Data source",
        ["upload", "sample"],
        format_func=lambda s: "Upload a CSV file" if s == "upload" else "Use the sample dataset",
        horizontal=True,
    )
    uploaded = None
    if data_src == "upload":
        uploaded = st.file_uploader("CSV file (WKT,name,description)", type=["csv"])

if data_src == "sample":
    source = io.BytesIO(SAMPLE_CSV.read_bytes())
else:
    source = uploaded

df = parse_csv(source)

with col_tile:
    tile_name = st.selectbox("Base map", list(TILE_LAYERS))

with col_filters:
    active = st.multiselect(
        "Layers to show",
        list(CATEGORY_CONFIG),
        default=list(CATEGORY_CONFIG),
        format_func=lambda c: LAYER_NAMES[c],
    )

with col_legend:
    legend = "".join(
        f'<div style="margin:2px 0;font-size:13px;">'
        f'<span style="display:inline-block;width:13px;height:13px;background:{cfg["color"]};'
        f'border-radius:2px;margin-right:6px;vertical-align:middle;"></span>{cfg["legend"]}</div>'
        for cfg in CATEGORY_CONFIG.values()
    )
    st.markdown(f"**Legend**<div>{legend}</div>", unsafe_allow_html=True)

alt_row_a, alt_row_b = st.columns([1, 2])
with alt_row_a:
    show_labels = st.checkbox("Show name labels on the map", value=False)
with alt_row_b:
    fetch_alt = st.checkbox(
        "Fetch altitude (Open-Elevation) for points",
        value=True,
        help="Uses the Z coordinate from 3D WKT (e.g. POINT Z (x y z)) when available; "
             "otherwise queries the free Open-Elevation API and caches the result.",
    )

alts = compute_altitudes(df, fetch_alt)

if df.empty:
    if source is None:
        st.info("No data loaded yet. Upload a CSV file (columns: **WKT, name, "
                "description**), or pick **Use the sample dataset** above to "
                "explore a real land configuration.")
    else:
        st.warning("No usable WKT features found in the file. "
                   "Check the CSV (columns WKT, name, description).")
    st.stop()

features, markers = build_features(df)
counts = {cat: len(features.get(cat, [])) + len(markers.get(cat, [])) for cat in CATEGORY_CONFIG}
st.caption(" | ".join(f"{LAYER_NAMES[c]}: {counts[c]}" for c in CATEGORY_CONFIG))

# ---------------- auto-generated output (after planner calculation) --------
categorized = categorize_df(df, alts)
processed_csv = processed_csv_string(categorized)

map_left, map_right = st.columns([3.4, 1.6])

with map_left:
    m = build_map(df, tile_name, active, show_labels, alts)
    result = st_folium(m, width="100%", height=640)

    clicked = result.get("last_object_clicked") if result else None
    if clicked:
        st.info(f"Clicked feature: {clicked}")

with map_right:
    st.subheader("Parsed data")
    st.dataframe(categorized, width="stretch", height=360)

    st.download_button(
        "Download categorized CSV",
        processed_csv,
        file_name="mymaps_categorized.csv",
        mime="text/csv",
    )

    with st.expander("CSV format help"):
        st.markdown(
            """
            | WKT | name | description |
            |-----|------|-------------|
            | `POINT (x y)` | Puit | Weel |
            | `LINESTRING (x y, x y, ...)` | R1 | Route |
            | `POLYGON ((x y, x y, ...))` | S1 | Sector |

            For altitude you can also use 3D coordinates, e.g.
            `POINT Z (x y z)`, `LINESTRING Z (...)` or `POLYGON Z ((...))`.
            Points without a Z are looked up online (Open-Elevation) when the
            "Fetch altitude" box is checked; results are cached in `elevations_cache.json`.

            Any instance of these words in name/description sets the category:
            **Land/terrain/perimetre**, **Sector/secteur**, **Zone**,
            **Route/trachee/tuyau/canal**, **Weel/well/puit/reservoir/basin**.
            Everything else becomes a Position or Structure. A lone unclassified
            polygon is treated as the land boundary.
            """
        )

st.divider()
render_planner_tab(df, fetch_alt)