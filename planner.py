"""Irrigation planning for the MyMaps app.

Pure computation + Streamlit rendering helpers. The planner:

  * estimates the land parcel (explicit "Land" feature, or the outline of all
    polygons in the CSV)
  * splits the land into sectors, each <= a max area (default 10 000 m2 = 1 ha)
  * splits every sector into N zones (default 3), each with one valve
  * sizes the basin from the tree count / spacing of the selected crop
    (e.g. olive trees at 4x4 m), plus water need per tree and autonomy days
  * uses the loaded point altitudes to estimate the slope, orient the
    trenches along contour lines, decide gravity vs pumped feed and
    therefore pick the layout with the lowest budget
  * draws 3 alternative trench/pipe configurations and compares their cost
"""

import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from shapely import wkt as swkt
from shapely.geometry import LineString, Point, Polygon, box, mapping
from shapely.ops import nearest_points, unary_union
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
ALT_CACHE_FILE = BASE_DIR / "elevations_cache.json"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

TILE_LAYERS = {
    "Google Satellite": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "Google Hybrid (labels)": {
        "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "Google Streets": {
        "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attr": "Google",
    },
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attr": "OpenStreetMap",
    },
}

LAND_KW = ("land", "terrain", "perimetre", "périmètre", "parcelle", "boundary")
BASIN_KW = ("basin", "reservoir", "bassin", "weel", "well", "puit")

CROPS = {
    "Olive 4x4 m": (4.0, 4.0, 30.0),
    "Citrus 5x5 m": (5.0, 5.0, 40.0),
    "Almond 6x6 m": (6.0, 6.0, 40.0),
    "Palm 8x8 m": (8.0, 8.0, 80.0),
    "Grapevine 2.5x1.5 m": (2.5, 1.5, 15.0),
    "Custom": (4.0, 4.0, 30.0),
}

CONFIG_STYLE = {
    "A": {"color": "#2e7d32", "label": "A - Gravity contour trenches"},
    "B": {"color": "#c2185b", "label": "B - Sector ring network"},
    "C": {"color": "#e65100", "label": "C - Minimum-pipe valve tree"},
}


# ---------------------------------------------------------------------------
# Altitude / elevation helpers (shared with app.py)
# ---------------------------------------------------------------------------
def fmt_alt(value) -> str:
    """Human readable altitude. Value can be float/None."""
    return "-" if value is None else f"{value:.1f} m"


def extract_z(g):
    """Return the Z coordinate for a 3D geometry, else None.

    For lines/polygons the mean of the vertex Z values is returned.
    """
    try:
        coords = list(g.coords)
    except NotImplementedError:
        coords = []
        try:
            if g.geom_type == "Polygon":
                coords = list(g.exterior.coords)
                for ring in g.interiors:
                    coords.extend(ring.coords)
            elif g.geom_type == "MultiPolygon":
                for poly in g.geoms:
                    coords.extend(poly.exterior.coords)
        except Exception:
            return None
    zs = [c[2] for c in coords if len(c) > 2]
    if not zs:
        return None
    return float(sum(zs) / len(zs))


def _load_alt_cache() -> dict:
    cache = getattr(_load_alt_cache, "_cache", None)
    if cache is None:
        cache = {}
        try:
            data = json.loads(ALT_CACHE_FILE.read_text(encoding="utf-8"))
            for key, val in data.items():
                try:
                    lat, lon = (float(p) for p in key.split(","))
                    cache[(lat, lon)] = float(val)
                except (ValueError, TypeError):
                    continue
        except Exception:
            cache = {}
        _load_alt_cache._cache = cache
    return cache


def _save_alt_cache(cache: dict):
    try:
        ALT_CACHE_FILE.write_text(
            json.dumps({f"{lat},{lon}": val for (lat, lon), val in cache.items()}, indent=0),
            encoding="utf-8",
        )
    except Exception:
        pass


def fetch_elevations(lat_lons, chunk: int = 50) -> dict:
    """Query the free Open-Elevation API for (lat, lon) tuples.

    Returns {(lat, lon): elevation_m} for the points that answered.
    """
    result = {}
    if not lat_lons:
        return result
    for i in range(0, len(lat_lons), chunk):
        part = lat_lons[i:i + chunk]
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in part]}
        try:
            resp = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for (lat, lon), r in zip(part, data.get("results", [])):
                try:
                    result[(lat, lon)] = float(r.get("elevation"))
                except (TypeError, ValueError):
                    continue
        except Exception:
            continue
    return result


def compute_altitudes(df: pd.DataFrame, fetch_online: bool) -> dict:
    """Map pandas index -> altitude in meters.

    Uses the WKT Z coordinate when available, otherwise (for POINT features)
    queries Open-Elevation and caches the results to disk.
    """
    alts = {}
    if df.empty:
        return alts

    cache = _load_alt_cache() if fetch_online else {}
    missing = []  # (index, (lat, lon))
    for idx, row in df.iterrows():
        g = row["geometry"]
        z = extract_z(g)
        if z is not None:
            alts[idx] = z
            continue
        if fetch_online and g.geom_type == "Point":
            key = (float(g.y), float(g.x))
            if key in cache:
                alts[idx] = cache[key]
            else:
                missing.append((idx, key))

    if missing and fetch_online:
        fresh = fetch_elevations([k for _, k in missing])
        for idx, key in missing:
            value = fresh.get(key)
            if value is not None:
                cache[key] = value
                alts[idx] = value
        _save_alt_cache(cache)

    return alts


def lookup_alts(points, fetch_online: bool = True) -> dict:
    """Elevation for (lon, lat) points via cache / Open-Elevation API.

    Returns {(lon, lat): elevation_m}.
    """
    out = {}
    if not points:
        return out
    cache = _load_alt_cache() if fetch_online else {}
    missing = []
    for pt in points:
        key = (pt[1], pt[0])  # (lat, lon)
        if key in cache:
            out[pt] = cache[key]
        else:
            missing.append((key, pt))
    if missing and fetch_online:
        fresh = fetch_elevations([k for k, _ in missing])
        for key, pt in missing:
            val = fresh.get(key)
            if val is not None:
                cache[key] = val
                out[pt] = val
        _save_alt_cache(cache)
    return out


# ---------------------------------------------------------------------------
# geodesic helpers (degrees <-> meters)
# ---------------------------------------------------------------------------
def meters_per_deg(poly_or_bounds):
    minx, miny, maxx, maxy = poly_or_bounds
    lat = math.radians((miny + maxy) / 2.0)
    return 111320.0 * math.cos(lat), 110574.0


def area_m2(poly):
    minx, miny, maxx, maxy = poly.bounds
    if minx == maxx or miny == maxy:
        return 0.0
    lon_scale, lat_scale = meters_per_deg(poly.bounds)
    return abs(poly.area) * lon_scale * lat_scale


def extents_m(poly):
    minx, miny, maxx, maxy = poly.bounds
    lon_scale, lat_scale = meters_per_deg(poly.bounds)
    return (maxx - minx) * lon_scale, (maxy - miny) * lat_scale


def dist_m(a, b):
    """Distance in meters between two (lon, lat) points."""
    lat = math.radians((a[1] + b[1]) / 2.0)
    x = (b[0] - a[0]) * 111320.0 * math.cos(lat)
    y = (b[1] - a[1]) * 110574.0
    return math.hypot(x, y)


def perimeter_m(poly):
    def seq(coords):
        return sum(dist_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))

    if poly.geom_type == "Polygon":
        total = seq(list(poly.exterior.coords))
        for ring in poly.interiors:
            total += seq(list(ring.coords))
        return total
    if poly.geom_type == "MultiPolygon":
        return sum(perimeter_m(p) for p in poly.geoms)
    return 0.0


# ---------------------------------------------------------------------------
# geometry splitting helpers
# ---------------------------------------------------------------------------
def flatten(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "LineString":
        return [geom]
    return []


def absorb_small(parts, min_area=80.0):
    """Merge slivers (< min_area m2) into their closest neighbour."""
    parts = [g for g in parts if g.geom_type == "Polygon"]
    guard = 0
    while guard < 500:
        guard += 1
        small = [g for g in parts if area_m2(g) < min_area]
        if not small:
            return parts
        smallest = min(small, key=area_m2)
        others = [g for g in parts if g is not smallest]
        if not others:
            return [g for g in parts if g is not smallest]
        nearest = min(others, key=lambda g: smallest.distance(g))
        merged = unary_union([smallest, nearest])
        if merged.geom_type == "Polygon":
            parts = [g for g in parts if g is not smallest and g is not nearest] + [merged]
        else:
            parts = [g for g in parts if g is not smallest]
    return parts


def split_poly(poly, dim, frac):
    """Cut a polygon with a vertical (dim='x') or horizontal (dim='y') line
    at a given fraction of its bounding box. Returns two polygons."""
    minx, miny, maxx, maxy = poly.bounds
    if dim == "x":
        cut = minx + frac * (maxx - minx)
        a = poly.intersection(box(minx, miny, cut, maxy))
        b = poly.intersection(box(cut, miny, maxx, maxy))
    else:
        cut = miny + frac * (maxy - miny)
        a = poly.intersection(box(minx, miny, maxx, cut))
        b = poly.intersection(box(minx, cut, maxx, maxy))
    return a, b


def split_to_max_area(poly, max_area, min_area=120.0, depth_max=14):
    """Recursively split a polygon so every piece is <= max_area m2."""
    out = []
    stack = [(poly, 0)]
    while stack:
        p, d = stack.pop()
        if p is None or p.is_empty:
            continue
        if area_m2(p) <= max_area or d >= depth_max:
            out.append(p)
            continue
        lon_m, lat_m = extents_m(p)
        dim = "x" if lon_m >= lat_m else "y"
        a, b = split_poly(p, dim, 0.5)
        for q in (*flatten(a), *flatten(b)):
            stack.append((q, d + 1))
    return absorb_small(out, min_area)


def split_into_parts(poly, n, min_area=60.0):
    """Split a polygon into ~n strips of equal bbox width/height."""
    if n <= 1 or poly is None or poly.is_empty:
        return [poly] if poly and not poly.is_empty else []
    minx, miny, maxx, maxy = poly.bounds
    lon_m, lat_m = extents_m(poly)
    dim = "x" if lon_m >= lat_m else "y"
    parts = []
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        if dim == "x":
            xa = minx + lo * (maxx - minx)
            xb = minx + hi * (maxx - minx)
            seg = poly.intersection(box(xa, miny, xb, maxy))
        else:
            ya = miny + lo * (maxy - miny)
            yb = miny + hi * (maxy - miny)
            seg = poly.intersection(box(minx, ya, maxx, yb))
        parts.extend(flatten(seg))
    parts = [g for g in parts if not g.is_empty]
    return absorb_small(parts, min_area)


def split_by_area(poly, dim, frac=0.5):
    """Cut a polygon so the 'lower' part holds ~frac of the total area.

    Binary-searches the cut position along the dominant axis (dim='x'/'y').
    Returns (lower_part, upper_part).
    """
    total = max(area_m2(poly), 1e-9)
    minx, miny, maxx, maxy = poly.bounds
    span = (maxx - minx) if dim == "x" else (maxy - miny)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        c = minx + mid * (maxx - minx) if dim == "x" else miny + mid * (maxy - miny)
        seg = (poly.intersection(box(minx, miny, c, maxy)) if dim == "x"
               else poly.intersection(box(minx, miny, maxx, c)))
        a = area_m2(seg) if not seg.is_empty else 0.0
        if a < total * frac:
            lo = mid
        else:
            hi = mid
    c = minx + hi * (maxx - minx) if dim == "x" else miny + hi * (maxy - miny)
    lower = (poly.intersection(box(minx, miny, c, maxy)) if dim == "x"
             else poly.intersection(box(minx, miny, maxx, c)))
    upper = (poly.intersection(box(c, miny, maxx, maxy)) if dim == "x"
             else poly.intersection(box(minx, c, maxx, maxy)))
    return lower, upper


def split_equal_areas(poly, n, dim="auto"):
    """Recursively cut a polygon into n strips of (nearly) equal area.

    dim: "auto" cuts along the longest bbox dimension, "x" produces vertical
    (north-south) strips, "y" produces horizontal (east-west) strips.
    """
    if n <= 1 or poly is None or poly.is_empty:
        return [poly] if poly and not poly.is_empty else []
    if dim == "auto":
        lon_m, lat_m = extents_m(poly)
        dim = "x" if lon_m >= lat_m else "y"
    left_n = n // 2
    lower, upper = split_by_area(poly, dim, left_n / n)
    if lower.is_empty or upper.is_empty:
        return split_into_parts(poly, n)
    return split_equal_areas(lower, left_n, dim) + split_equal_areas(upper, n - left_n, dim)


def split_to_band(poly, lo_m2, hi_m2, dim="auto"):
    """Split a polygon into sectors whose area falls inside [lo_m2, hi_m2].

    Sectors are equal-area strips so every one lands inside the band.
    dim is forwarded to split_equal_areas (see there).
    """
    if poly is None or poly.is_empty:
        return []
    total = area_m2(poly)
    if total <= 0 or total <= hi_m2:
        return [poly]

    lo_m2, hi_m2 = sorted((float(lo_m2), float(hi_m2)))
    target = (lo_m2 + hi_m2) / 2.0
    n = max(1, int(round(total / target)))

    direct = split_equal_areas(poly, n, dim)
    if all(lo_m2 <= area_m2(p) <= hi_m2 for p in direct):
        return absorb_small(direct, 60.0)

    candidates = [split_equal_areas(poly, t, dim)
                  for t in (n, n + 1, n - 1, n + 2, n - 2) if t >= 1]
    for parts in candidates:
        if parts and all(lo_m2 <= area_m2(p) <= hi_m2 for p in parts):
            return absorb_small(parts, 60.0)
    best = min(candidates, key=lambda ps: sum(abs(area_m2(p) - target) for p in ps))
    return absorb_small(best, 60.0)


def get_land_polygon(df: pd.DataFrame):
    """Estimate the land parcel outline from the CSV features."""
    if df is None or df.empty:
        return None

    explicit = []
    all_polys = []
    for row in df.itertuples():
        g = row.geometry
        if g.geom_type not in ("Polygon",):
            continue
        all_polys.append(g)
        if any(k in f"{row.name} {row.description}".lower() for k in LAND_KW):
            explicit.append(g)

    if explicit:
        parts = flatten(unary_union(explicit))
        if parts:
            return max(parts, key=area_m2)
    if all_polys:
        parts = flatten(unary_union(all_polys))
        if parts:
            return max(parts, key=area_m2)
    return None


def existing_basin(df: pd.DataFrame):
    for row in df.itertuples():
        g = row.geometry
        if g.geom_type == "Polygon" and any(
            k in f"{row.name} {row.description}".lower() for k in BASIN_KW
        ):
            return g
    return None


def land_alt_points(df: pd.DataFrame):
    """(lon, lat, alt) samples across the field for slope estimation."""
    rows = []
    for row in df.itertuples():
        g = row.geometry
        try:
            coords = list(g.coords)
        except NotImplementedError:
            coords = list(g.exterior.coords) if g.geom_type == "Polygon" else []
        for c in coords:
            if len(c) >= 3:
                rows.append((c[0], c[1], float(c[2])))
    return rows


# ---------------------------------------------------------------------------
# slope / trench helpers
# ---------------------------------------------------------------------------
def _solve3(a11, a12, a13, b1, a21, a22, a23, b2, a31, a32, a33, b3):
    import numpy as np  # available with pandas/streamlit

    A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=float)
    B = np.array([b1, b2, b3], dtype=float)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None


def fit_down_vector(alt_pts):
    """Best-fit plane z = a*x + b*y + c over (lon, lat, alt) samples.

    Returns the unit downhill direction (in lon/lat degrees) or None if the
    sampled elevation is too unreliable.
    """
    pts = [p for p in alt_pts if p[2] is not None]
    if len(pts) < 3:
        # with 1 or 2 points fall back to the strongest pair
        return None
    lon_scale, lat_scale = meters_per_deg(
        (min(p[0] for p in pts), min(p[1] for p in pts),
         max(p[0] for p in pts), max(p[1] for p in pts))
    )
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    X = [(p[0] - mx) * lon_scale for p in pts]
    Y = [(p[1] - my) * lat_scale for p in pts]
    Z = [p[2] for p in pts]
    n = len(pts)
    sxx = sum(x * x for x in X)
    syy = sum(y * y for y in Y)
    sxy = sum(x * y for x, y in zip(X, Y))
    sxz = sum(x * z for x, z in zip(X, Z))
    syz = sum(y * z for y, z in zip(Y, Z))
    sol = _solve3(sxx, sxy, sum(X), sxz,
                  sxy, syy, sum(Y), syz,
                  sum(X), sum(Y), n, sum(Z))
    if sol is None:
        return None
    a, b, _ = sol
    gx, gy = -a / lon_scale, -b / lat_scale  # downhill in degrees space
    norm = math.hypot(gx, gy)
    if norm < 1e-9:
        return None
    return gx / norm, gy / norm


def contour_dir(vh):
    """Direction perpendicular to the downhill vector (degrees space)."""
    if vh is None:
        return (1.0, 0.0)  # default East-West contour
    return (vh[1], -vh[0])


def clip_line(poly, p0, p1):
    """Intersect a polygon with the segment p0->p1 (infinite line)."""
    line = LineString([p0, p1])
    inter = poly.intersection(line)
    out = []
    if inter.geom_type == "LineString":
        out = [list(inter.coords)]
    elif inter.geom_type == "MultiLineString":
        out = [list(g.coords) for g in inter.geoms]
    return out


def zone_trench(zone, u):
    """Contour trench through the zone centroid along direction u."""
    minx, miny, maxx, maxy = zone.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    L = (maxx - minx + maxy - miny) * 1.5
    p0 = (cx - u[0] * L, cy - u[1] * L)
    p1 = (cx + u[0] * L, cy + u[1] * L)
    lines = clip_line(zone, p0, p1)
    if lines:
        return lines
    return [[(minx, cy), (maxx, cy)]]


def zone_furrows(zone, vh, n=2):
    """Short downhill furrows (parallel to vh) for a zone."""
    if vh is None:
        vh = (0.0, -1.0)  # assume south is downhill
    u = (vh[1], -vh[0])
    minx, miny, maxx, maxy = zone.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    L = (maxx - minx + maxy - miny) * 1.2
    out = []
    for k in range(n):
        f = (k + 1) / (n + 1)
        ox = u[0] * f * L * 0.4
        oy = u[1] * f * L * 0.4
        p0 = (cx + ox - vh[0] * L, cy + oy - vh[1] * L)
        p1 = (cx + ox + vh[0] * L, cy + oy + vh[1] * L)
        out.extend(clip_line(zone, p0, p1))
    return out


def centroid_pt(poly):
    p = poly.centroid
    return (p.x, p.y)


# ---------------------------------------------------------------------------
# MST (minimum spanning tree) for pipe-length estimates
# ---------------------------------------------------------------------------
def mst(points):
    """Prim's MST over (lon, lat) points -> (total meters, edges [(i, j)])."""
    n = len(points)
    if n <= 1:
        return 0.0, []
    used = [False] * n
    used[0] = True
    edges = []
    total = 0.0
    for _ in range(n - 1):
        best_i, best_j, best_d = -1, -1, float("inf")
        for i in range(n):
            if used[i]:
                continue
            for j in range(n):
                if used[j]:
                    d = dist_m(points[i], points[j])
                    if d < best_d:
                        best_d, best_i, best_j = d, i, j
        if best_i < 0:
            break
        edges.append((best_i, best_j))
        total += best_d
        used[best_i] = True
    return total, edges


# ---------------------------------------------------------------------------
# basin sizing & cost engine
# ---------------------------------------------------------------------------
def basin_design(land_area_m2, spacing, liters_per_tree_day, autonomy_days, depth_m):
    sx, sy = spacing
    if sx <= 0 or sy <= 0:
        return None
    trees = int(land_area_m2 / (sx * sy))
    daily_m3 = trees * liters_per_tree_day / 1000.0
    volume_m3 = daily_m3 * autonomy_days
    footprint_m2 = volume_m3 / max(depth_m, 0.5)
    return {
        "trees": trees,
        "daily_m3": daily_m3,
        "volume_m3": volume_m3,
        "footprint_m2": footprint_m2,
        "spacing": (sx, sy),
    }


def pump_power_kw(flow_m3s, head_m, efficiency=0.7):
    if head_m <= 0:
        return 0.0
    return (flow_m3s * 9.81 * head_m) / (efficiency)


def pump_flow_m3s(daily_m3, pump_hours_day):
    if pump_hours_day <= 0:
        return 0.0
    return daily_m3 / (pump_hours_day * 3600.0)


def build_configs(land, sectors, zones, basin_pt, u, vh):
    """Build the 3 pipe/trench configuration options.

    zones: list of dicts {poly, sector_idx, valve(idx,lon,lat), label}
    """
    sec_centroids = [centroid_pt(s) for s in sectors]
    valve_pts = [z["valve"] for z in zones]
    n_valves = len(zones)

    # --- A: gravity contour headers + zone trenches + furrows
    nodes_a = [basin_pt] + sec_centroids
    header_a, edges_a = mst(nodes_a)
    trench_m = 0.0
    furrow_m = 0.0
    lines_a = []
    for start, end in edges_a:
        lines_a.append([nodes_a[start], nodes_a[end]])
    for z in zones:
        tr_lines = zone_trench(z["poly"], u)
        for ln in tr_lines:
            if len(ln) < 2:
                continue
            lines_a.append(ln)
            trench_m += sum(dist_m(ln[i], ln[i + 1]) for i in range(len(ln) - 1))
        for fr in zone_furrows(z["poly"], vh, n=2):
            if len(fr) < 2:
                continue
            lines_a.append(fr)
            furrow_m += sum(dist_m(fr[i], fr[i + 1]) for i in range(len(fr) - 1))
    length_a = header_a + trench_m + furrow_m

    # --- B: ring around each sector + links from valves to the ring
    lines_b = []
    ring_m = 0.0
    link_m = 0.0
    for s in sectors:
        hull = s.convex_hull
        ring_coords = list(hull.exterior.coords)
        lines_b.append(ring_coords)
        ring_m += perimeter_m(hull)
        ring_geom = hull.boundary
        for z in zones:
            if z["sector_idx"] == sectors.index(s):
                p1, p2 = nearest_points(Point(z["valve"]), ring_geom)
                lines_b.append([(p1.x, p1.y), (p2.x, p2.y)])
                link_m += dist_m((p1.x, p1.y), (p2.x, p2.y))
    length_b = ring_m + link_m

    # --- C: minimum-pipe tree (MST) linking basin + every valve
    nodes_c = [basin_pt] + valve_pts
    length_c, edges_c = mst(nodes_c)
    lines_c = []
    for start, end in edges_c:
        lines_c.append([nodes_c[start], nodes_c[end]])

    return [
        {"key": "A", "label": CONFIG_STYLE["A"]["label"], "color": CONFIG_STYLE["A"]["color"],
         "length_m": length_a, "n_valves": n_valves, "lines": lines_a,
         "desc": "Gravity-fed: one header from the basin to every sector, one "
                 "contour trench per zone with short downhill furrows. Cheapest "
                 "pipe length and zero pumping if the basin is uphill."},
        {"key": "B", "label": CONFIG_STYLE["B"]["label"], "color": CONFIG_STYLE["B"]["color"],
         "length_m": length_b, "n_valves": n_valves, "lines": lines_b,
         "desc": "Robust loop: a ring around each sector fed with multiple "
                 "links. Highest pipe length and cost, best pressure "
                 "uniformity and redundancy."},
        {"key": "C", "label": CONFIG_STYLE["C"]["label"], "color": CONFIG_STYLE["C"]["color"],
         "length_m": length_c, "n_valves": n_valves, "lines": lines_c,
         "desc": "Balanced: the minimum-possible pipe length (spanning tree) "
                 "from the basin to every sector valve. Middle-ground budget."},
    ]


# ---------------------------------------------------------------------------
# planner map
# ---------------------------------------------------------------------------
def build_planner_map(land, sectors, zones, valves, basin_pt, basin_alt,
                      configs, tile_name, valve_alts, show_sectors_only=False):
    center = [(land.bounds[1] + land.bounds[3]) / 2, (land.bounds[0] + land.bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=18, tiles=None, control_scale=True)
    for name, cfg in TILE_LAYERS.items():
        folium.TileLayer(tiles=cfg["url"], attr=cfg["attr"], name=name,
                         show=(name == tile_name)).add_to(m)
    folium.FitBounds([[land.bounds[1], land.bounds[0]], [land.bounds[3], land.bounds[2]]]).add_to(m)

    # land outline
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "geometry": mapping(land),
                        "properties": {"area_m2": area_m2(land)}}]}
    folium.GeoJson(
        fc, name="Land", style_function=lambda f: {
            "color": "#d73027", "weight": 4, "opacity": 1, "fill": False},
        tooltip=folium.GeoJsonTooltip(fields=["area_m2"], aliases=["Area m2"],
                                      labels=True, sticky=True),
    ).add_to(m)

    # sectors
    fc_s = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(s),
         "properties": {"sector": i + 1, "area_m2": round(area_m2(s))}}
        for i, s in enumerate(sectors)]}
    folium.GeoJson(
        fc_s, name="Sectors", style_function=lambda f: {
            "color": "#f07d00", "weight": 2, "fillColor": "#f07d00",
            "fillOpacity": 0.10, "dashArray": "5,5"},
        tooltip=folium.GeoJsonTooltip(fields=["sector", "area_m2"],
                                       aliases=["Sector", "Area m2"], sticky=True),
    ).add_to(m)

    if show_sectors_only:
        folium.LayerControl(collapsed=True).add_to(m)
        return m

    # zones
    fc_z = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(z["poly"]),
         "properties": {"zone": z["label"], "valve": z["valve"][0]}}
        for z in zones]}
    folium.GeoJson(
        fc_z, name="Zones (3 / sector)", style_function=lambda f: {
            "color": "#2b83ba", "weight": 1.5, "fillColor": "#2b83ba",
            "fillOpacity": 0.18},
        tooltip=folium.GeoJsonTooltip(fields=["zone"], aliases=["Zone"], sticky=True),
    ).add_to(m)

    # valves
    vg = folium.FeatureGroup(name="Valves")
    for i, z in enumerate(zones, 1):
        a = valve_alts.get(z["valve"])
        tip = f"V{i} - {z['label']}" + (f" | Alt: {a:.1f} m" if a is not None else "")
        folium.CircleMarker(
            location=[z["valve"][1], z["valve"][0]], radius=8, color="#1565c0",
            weight=2, fill=True, fill_color="#42a5f5", fill_opacity=0.9,
            tooltip=tip, popup=tip,
        ).add_to(vg)
    vg.add_to(m)

    # basin
    ba = "not available"
    if basin_alt is not None:
        ba = f"{basin_alt:.1f} m"
    folium.Marker(
        location=[basin_pt[1], basin_pt[0]], icon=folium.Icon(color="red", icon="tint", prefix="fa"),
        tooltip=f"Basin (gravity feed point) | Alt {ba}",
    ).add_to(m)

    # config polylines
    for cfg in configs:
        g = folium.FeatureGroup(name=cfg["label"])
        for coords in cfg["lines"]:
            if len(coords) < 2:
                continue
            folium.PolyLine([(lat, lon) for lon, lat in coords],
                            color=cfg["color"], weight=3, opacity=0.9).add_to(g)
        g.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ---------------------------------------------------------------------------
# plan export to CSV (same format as the input: WKT,name,description)
# ---------------------------------------------------------------------------
def _wtk(geom, precision: int = 6) -> str:
    try:
        return swkt.dumps(geom, rounding_precision=precision, trim=True)
    except Exception:
        return geom.wkt


def plan_to_csv(land, sectors, zones, config=None, valve_alts=None) -> str:
    """Serialize the final plan into a CSV with WKT,name,description columns.

    config: one of the options returned by build_configs() whose polyline
    layout should be included, or None to export geometry only.
    """
    valve_alts = valve_alts or {}
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["WKT", "name", "description"])

    w.writerow([_wtk(land), "Land", "Land"])
    for i, s in enumerate(sectors, 1):
        w.writerow([_wtk(s), f"S{i}", "Sector"])
    for z in zones:
        w.writerow([_wtk(z["poly"]), z["label"], "Zone"])

    for vi, z in enumerate(zones, 1):
        a = valve_alts.get(z["valve"])
        desc = "Valve" + (f" | Alt {a:.1f} m" if a is not None else "")
        w.writerow([_wtk(Point(z["valve"])), f"V{vi}", desc])

    if config is not None:
        for ji, line in enumerate(config["lines"], 1):
            if len(line) < 2:
                continue
            w.writerow([_wtk(LineString(line)), f"{config['key']}-{ji}",
                        f"Trachee | option {config['key']}"])
    return out.getvalue()


# ---------------------------------------------------------------------------
# Streamlit planner tab
# ---------------------------------------------------------------------------
def eur(x):
    return f"{x:,.0f} €"


def _auto_save_plan(plan_csv: str):
    """Write the finished plan (land, sectors, zones, valves, trenches) to output/.

    Writes a new timestamped file once per distinct plan content so repeated
    widget tweaks don't spam the output folder; returns the file path or None.
    """
    key = hashlib.md5(plan_csv.encode("utf-8")).hexdigest()
    if st.session_state.get("auto_plan_key") == key:
        return st.session_state.get("auto_plan_path")
    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = out_dir / name
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(plan_csv)
    st.session_state.auto_plan_key = key
    st.session_state.auto_plan_path = str(path)
    return str(path)


def render_planner_tab(df: pd.DataFrame, fetch_alt: bool):
    st.subheader("Irrigation planner")
    st.caption("Sectors are equal strips sized inside the chosen area band "
               "(default 9 000 - 11 000 m2), zones = valves, basin size follows "
               "the tree count - plain language: dodge the pipe, follow the slope.")

    land = get_land_polygon(df)
    if land is None:
        st.info("No land outline found. Add a polygon with 'land/terrain/perimetre' "
                "in its name/description, or any polygons to outline the field.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        crop = st.selectbox("Crop", list(CROPS), index=0)
        sx, sy, lpd = CROPS[crop]
        sp = st.slider("Tree spacing (m)", 1.0, 12.0, (min(sx, sy), max(sx, sy)), step=0.5)
    with c2:
        liters = st.number_input("Water per tree per day (L)", 1.0, 200.0,
                                 float(lpd), step=1.0)
        autonomy = st.number_input("Basin autonomy (days)", 1, 30, 3)
        depth = st.number_input("Basin depth (m)", 0.5, 10.0, 3.0, step=0.5)
    with c3:
        sec_min = st.number_input("Sector min area (m2)", 1000, 100000, 9000, step=500)
        sec_max = st.number_input("Sector max area (m2)", 1000, 200000, 11000, step=500)
        sec_min, sec_max = sorted((sec_min, sec_max))
        zones_per = st.slider("Zones per sector (valves)", 2, 6, 3)
        pump_hrs = st.number_input("Pumping hours per day", 1, 24, 10)
    with c4:
        pipe_cost = st.number_input("Pipe cost (€/m)", 0.5, 20.0, 2.0, step=0.1)
        valve_cost = st.number_input("Valve cost (€)", 5.0, 500.0, 30.0, step=5.0)
        pump_cost = st.number_input("Pump cost (€/kW installed)", 100.0, 2000.0, 500.0, step=50.0)
        basin_cost = st.number_input("Basin excavation (€/m3)", 1.0, 50.0, 4.0, step=0.5)

    # ---------------- geometry ----------------
    land_area = area_m2(land)
    sector_configs = [
        {"key": "balanced", "label": "Balanced (cut along the longest side)",
         "sectors": split_to_band(land, sec_min, sec_max)},
        {"key": "rows", "label": "Rows (horizontal strips)",
         "sectors": split_to_band(land, sec_min, sec_max, dim="y")},
        {"key": "columns", "label": "Columns (vertical strips)",
         "sectors": split_to_band(land, sec_min, sec_max, dim="x")},
    ]
    sectors = sector_configs[0]["sectors"]
    n_sectors = len(sectors)
    zones = []
    for si, s in enumerate(sectors):
        pieces = split_into_parts(s, zones_per)
        for zi, pz in enumerate(pieces):
            sz = pz.area
            zname = f"S{si + 1}Z{zi + 1}"
            zones.append({"poly": pz, "sector_idx": si, "zone_idx": zi,
                          "label": zname, "valve": centroid_pt(pz), "area": area_m2(pz)})
    valve_pts = [z["valve"] for z in zones]

    # ---------------- altitude / slope ----------------
    alt_pts = land_alt_points(df)
    if not alt_pts and fetch_alt:
        # sample the field outline so we can estimate the slope
        coords = list(land.exterior.coords)
        step = max(1, len(coords) // 12)
        alt_pts = [(c[0], c[1], None) for c in coords[::step]]
    valve_alts = lookup_alts(valve_pts, fetch_alt)
    known_vals = [a for a in valve_alts.values() if a is not None]

    slope_samples = [(p[0], p[1], p[2]) for p in alt_pts]
    if known_vals:
        slope_samples += [(pt[0], pt[1], v) for pt, v in valve_alts.items() if v is not None]
    vh = fit_down_vector(slope_samples)
    u = contour_dir(vh)

    # basin placement: highest point so water can flow by gravity
    land_z = extract_z(land)
    if land_z is not None:
        basin_alt = land_z
        basin_pt = centroid_pt(land)
    else:
        candidates = dict(valve_alts)
        ex = existing_basin(df)
        if ex is not None:
            candidates[("existing_basin", 0.0)] = extract_z(ex)
        if candidates:
            best = max(candidates.items(), key=lambda kv: kv[1] if kv[1] is not None else -1e9)
            if best[1] is not None:
                basin_pt = best[0] if isinstance(best[0], tuple) and len(best[0]) == 2 else centroid_pt(land)
                basin_alt = best[1]
            else:
                basin_pt, basin_alt = centroid_pt(land), None
        else:
            basin_pt, basin_alt = centroid_pt(land), None

    # ---------------- basin sizing ----------------
    design = basin_design(land_area, (sp[0], sp[1]), liters, autonomy, depth)
    ex_basin = existing_basin(df)

    # ---------------- configs & cost ----------------
    configs = build_configs(land, sectors, zones, basin_pt, u, vh)

    max_valve_alt = max(known_vals) if known_vals else None
    if max_valve_alt is None or basin_alt is None:
        head_m = None
        pump_kw_disp = None
    else:
        head_m = max(0.0, max_valve_alt - basin_alt + 3.0)
    if design:
        flow = pump_flow_m3s(design["daily_m3"], pump_hrs)
        if head_m is not None:
            pump_kw = pump_power_kw(flow, head_m)
            pump_eur = pump_kw * pump_cost
        else:
            pump_kw, pump_eur = None, 0.0
    else:
        pump_kw, pump_eur = None, 0.0

    basin_eur = design["volume_m3"] * basin_cost if design else 0.0

    table = []
    for cfg in configs:
        pipe_eur = cfg["length_m"] * pipe_cost
        valves_eur = cfg["n_valves"] * valve_cost
        cfg["pipe_eur"] = pipe_eur
        cfg["valves_eur"] = valves_eur
        cfg["total"] = pipe_eur + valves_eur + pump_eur + basin_eur
        table.append({
            "Option": cfg["key"],
            "Description": cfg["label"],
            "Pipe (m)": round(cfg["length_m"]),
            "Valves": cfg["n_valves"],
            "Pump (kW)": round(pump_kw, 2) if pump_kw is not None else None,
            "Pipe cost": pipe_eur,
            "Valve cost": valves_eur,
            "Pump cost": pump_eur,
            "Basin cost": basin_eur,
            "Total": cfg["total"],
        })

    # ---------------- layout ----------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Land area", f"{land_area:,.0f} m2", f"{land_area / 10000:.2f} ha")
    m2.metric("Sectors (in band)", f"{n_sectors}")
    m3.metric("Zones / valves", f"{len(zones)}")
    m4.metric("Trees (%sx%s m)" % (sp[0], sp[1]),
              f"{design['trees']:,}" if design else "-")

    b1, b2, b3, b4 = st.columns(4)
    if design:
        b1.metric("Water / day", f"{design['daily_m3']:.1f} m3")
        b2.metric("Basin volume", f"{design['volume_m3']:.0f} m3")
        b3.metric("Basin footprint", f"{design['footprint_m2']:,.0f} m2")
        if ex_basin is not None:
            b4.metric("Existing basin", f"{area_m2(ex_basin):,.0f} m2",
                      f"needs {design['footprint_m2'] - area_m2(ex_basin):+,.0f} m2")
        else:
            b4.metric("Existing basin", "none",
                      f"{design['footprint_m2']:+,.0f} m2 to build")
    else:
        b1.metric("Water / day", "-")
        b2.metric("Basin volume", "-")
        b3.metric("Basin footprint", "-")
        b4.metric("Existing basin", "-")

    note = []
    if head_m is not None:
        if head_m <= 0:
            note.append("Gravity feed possible: every valve is below the basin -> "
                        "no pump needed (lowest budget).")
        else:
            note.append(f"Basin is {head_m:.0f} m below the highest valve -> a pump "
                        f"({pump_kw:.1f} kW) is required.")
    else:
        note.append("Fetch point altitudes to compute pumping needs (slope follow).")
    if vh is not None:
        note.append("Trenches are drawn along contour lines estimated from the "
                    "loaded point altitudes.")
    st.caption(" ".join(note))

    st.subheader("Sector configurations")
    sec_tile = st.selectbox("Sector base map", list(TILE_LAYERS), key="sector_tile")
    tabs = st.tabs([f"{cfg['label']} — {len(cfg['sectors'])} sectors"
                    for cfg in sector_configs])
    for tab, cfg in zip(tabs, sector_configs):
        with tab:
            cfg_df = pd.DataFrame([{
                "Sector": f"S{i+1}",
                "Area (m\u00b2)": round(area_m2(s)),
                "Area (ha)": round(area_m2(s) / 10000, 3),
                "Share (%)": round(100 * area_m2(s) / land_area, 2) if land_area else 0,
            } for i, s in enumerate(cfg["sectors"])])
            st.dataframe(cfg_df, width="stretch")
            cfg_map = build_planner_map(
                land, cfg["sectors"], [], [], basin_pt, basin_alt,
                [], sec_tile, {}, show_sectors_only=True)
            st_folium(cfg_map, width="100%", height=480)
    st.divider()

    # ---- costs & comparison ----
    dfc = pd.DataFrame(table)
    dfc = dfc.set_index("Option")
    st.subheader("Budget comparison (3 trench configurations)")
    st.dataframe(dfc.style.format({
        "Pipe (m)": "{:,.0f}", "Pump (kW)": "{:.2f}",
        "Pipe cost": eur, "Valve cost": eur, "Pump cost": eur,
        "Basin cost": eur, "Total": eur}, na_rep="-"), width="stretch")

    st.bar_chart(dfc["Total"])

    best = min(table, key=lambda r: r["Total"])
    st.success(
        f"**Lowest budget: Option {best['Option']}** ({best['Description']}) at "
        f"**{eur(best['Total'])}**. Total pipe {best['Pipe (m)']:,} m, "
        f"{best['Valves']} valves."
        + (" Gravity feed - no pump. " if head_m is not None and head_m <= 0 else " ")
        + "Trenches follow the contour computed from your point-level data."
    )

    exp_x, exp_y = st.columns([1, 2.4])
    with exp_x:
        exp_opt = st.selectbox(
            "Trench layout in export",
            ["Best (lowest budget)", "Option A", "Option B", "Option C", "None"],
            index=0,
            key="export_trench_opt",
        )
    key_map = {"Option A": "A", "Option B": "B", "Option C": "C"}
    if exp_opt.startswith("Best"):
        chosen_key = best["Option"]
    elif exp_opt == "None":
        chosen_key = None
    else:
        chosen_key = key_map[exp_opt]
    chosen_cfg = next((c for c in configs if c["key"] == chosen_key), None) if chosen_key else None
    plan_csv = plan_to_csv(land, sectors, zones, chosen_cfg, valve_alts)
    plan_path = _auto_save_plan(plan_csv)
    with exp_y:
        st.download_button(
            "Export final plan to CSV (input format)",
            plan_csv,
            file_name="irrigation_plan.csv",
            mime="text/csv",
        )
        st.caption(f"{n_sectors} sectors, {len(zones)} zones and valves"
                   + (" plus the trench polyline layout" if exp_opt != "None" else "")
                   + " - same WKT, name, description columns as the input CSV.")
        if plan_path:
            st.caption(f"Auto-generated after calculation: `{Path(plan_path).name}` "
                       f"(land, {n_sectors} sectors, {len(zones)} zones/valves"
                       + (" and trench layout" if exp_opt != "None" else "")
                       + ") saved to `output/`.")

    with st.expander("How the three trench options differ"):
        for cfg in configs:
            st.markdown(f"- **{cfg['label']}** (pipe ≈ {cfg['length_m']:,.0f} m, "
                        f"{cfg['n_valves']} valves) — {cfg['desc']}")