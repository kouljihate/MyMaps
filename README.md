# MyMaps

Streamlit app to visualize land parcels, sectors, zones and points of interest
from a CSV file on an interactive map — and to plan an irrigation layout over
that land.

## Features

- **CSV import** (`WKT, name, description` columns) via upload or built-in sample dataset.
- **Auto classification** of every feature into layers:
  Land boundary, Sector, Zone, Route/pipe, Water, Position/structure.
- **Interactive maps** backed by folium with Google Satellite / Google Hybrid /
  Google Streets / OpenStreetMap tiles.
- **Labels on the map** and a live **legend**.
- **Altitude handling**: uses the Z coordinate from 3D WKT when present,
  otherwise queries the free [Open-Elevation](https://open-elevation.com/) API
  and caches results in `elevations_cache.json`.
- **Processed table + download**: see each feature's category and altitude, and
  download the categorized CSV.
- **Irrigation planner** (note: estimation tool, not engineering design):
  - estimates the land parcel from the CSV polygons
  - splits the land into sectors sized inside a chosen area band (default 1 ha)
  - splits each sector into zones, each with one valve
  - sizes the basin from crop spacing, water need per tree and autonomy days
  - estimates slope from the loaded point altitudes to orient trenches along
    contour lines and decide gravity vs pumped feed
  - compares **3 trench/pipe configurations** (gravity contour trenches,
    sector ring network, minimum-pipe valve tree) and their budget
  - exports the final plan back to the same CSV format

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
streamlit>=1.30
streamlit-folium>=0.16
folium>=0.15
shapely>=2.0
pandas>=2.0
requests>=2.31
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then either pick **Use the sample dataset** or upload a CSV.

### CSV format

| WKT | name | description |
|-----|------|-------------|
| `POINT (x y)` | Puit | Weel |
| `LINESTRING (x y, x y, ...)` | R1 | Route |
| `POLYGON ((x y, x y, ...))` | S1 | Sector |

Coordinates are longitude / latitude degrees. You can also use 3D coordinates
(e.g. `POINT Z (x y z)`) — the Z is used as the altitude.

Any of these words in `name` or `description` set the category:

- **Land** — `land`, `terrain`, `perimetre` / `périmètre`, `parcelle`, `boundary`
- **Sector** — `sector`, `secteur`
- **Zone** — `zone`
- **Water** — `weel`, `well`, `puit`, `water`, `source`, `reservoir`, `basin`
- **Route** — `route`, `trachee` / `tranchée` / `tranchee`, `pipe`, `tuyau`, `line`, `canal`

Everything else becomes a Position/structure; a lone unclassified polygon is
assumed to be the land boundary.

## Project structure

```
app.py               Streamlit UI: CSV import, map layers, categorized table
planner.py           Irrigation planner: sectors, zones, basin sizing, cost comparison
sample.csv           Sample land configuration (a Moroccan plot near Mtarnagha)
requirements.txt     Python dependencies
elevations_cache.json  Altitude cache written at runtime (optional, regenerated)
input/               Your own CSVs (optional)
output/              Auto-generated plan CSVs written by the planner
```

## Notes

- The irrigation planner changes widget values live; the last computed plan is
  auto-saved as a timestamped CSV under `output/`.
- Trench layouts exported back to CSV use the same `WKT, name, description`
  columns as the input, so they can be re-imported into the map.