
# Shark Foraging Hotspots & Movement — Run Guide

This repo-like folder contains a **working pipeline** and **two lightweight dashboards**
to explore shark movement in relation to **eddies, fronts, shelf break, and MLD** using
**NASA-like fields** (SWOT SSH, MODIS/PACE chlorophyll, SST). You can run everything
**offline** with the bundled synthetic data or swap in **real NASA products**.

## Contents

- **Notebooks**
  - `Shark_Habitat_Model_Notebook.ipynb` — base version (synthetic demo, full pipeline).
  - `Shark_Habitat_Model_v2.ipynb` — expanded drivers (eddy polarity, EKE, |∇SST|, distances, bathymetry, MLD) + updated model/forecast.

- **Apps**
  - `app_streamlit.py` — interactive model fitting & forecasting (upload tag & fields, choose region/steps, draw maps & tracks).
  - `bokeh_app/main.py` — minimal map explorer (quick layer checks).

- **Data (demo)**
  - `env_fields_v2.npz` — synthetic environment bundle (SSH, U, V, SST, CHL, EKE, |∇SST|, masks, distances, bathymetry, MLD).
  - `example_tag_data.csv` — synthetic tag track (`time, lon, lat, depth_m`).

- **Utilities**
  - `requirements.txt` — Python packages.
  - `build_env_npz_from_netcdf.py` — helper to construct a NPZ bundle from *real* NetCDF files (SWOT SSH, SST, CHL).

---

## 1) Environment setup

```bash
# Recommended: use a fresh virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r /mnt/data/requirements.txt
```

> If you will convert NetCDF to NPZ, the script uses `xarray` and `netCDF4` (already in requirements).

---

## 2) Run the dashboards

### Streamlit (full workflow: fit + forecast)
```bash
streamlit run /mnt/data/app_streamlit.py
```
- **Sidebar** → upload:
  - **Tag CSV** with columns: `time, lon, lat[, depth_m]`
  - **Env NPZ** (or keep the default `env_fields_v2.npz`)
- Pick **region**, **forecast steps** (6h per step), **ensemble size**, **step length**.
- Click **Run forecast** → the app will:
  1. Fit the **integrated step-selection** (logistic) model on your tag.
  2. Compute an **HSI** from expanded drivers.
  3. Run an ensemble **forecast** (currents + HSI bias).
  4. Draw selected base layer (SSH, SST, CHL, EKE, |∇SST|, BATHY, MLD, or computed HSI) with observed & predicted tracks.

### Bokeh (light viewer)
```bash
bokeh serve --show /mnt/data/bokeh_app
```
- Choose layer, set lon/lat ranges, plot the sample track + simple forecasts.
- Use Streamlit for the full model & forecasting flow.

---

## 3) Run the notebooks

Open in Jupyter/Lab and run all cells:

- `/mnt/data/Shark_Habitat_Model_Notebook.ipynb` (base pipeline)
- `/mnt/data/Shark_Habitat_Model_v2.ipynb` (expanded drivers + distances)

Each notebook is fully documented, uses the same environment bundle, and shows how to **swap in real NASA fields** (see Section 4).

---

## 4) Using **real NASA data** with the apps/notebooks

All tools expect a compact **NPZ bundle** with fields on **one lon/lat grid**.
You can create it with the provided script: `build_env_npz_from_netcdf.py`.

### 4.1 Expected NPZ keys
The apps/notebooks read these keys from the NPZ file:
```
lons, lats, SSH, U, V, SPEED, SST, CHL,
W, EDDY_CORE_MASK, EDDY_EDGE_MASK, WARM_MASK,
EKE, SST_FRONT, BATHY, SHELF_MASK, MLD,
DIST_TO_CORE_M, DIST_TO_EDGE_M, DIST_TO_SHELF_M,
dx_m, dy_m, f, g, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
```
> You do **not** need to provide every key; the script computes the derived ones
> (U, V, EKE, |∇SST|, Okubo–Weiss, eddy masks, distances) for you.

### 4.2 Prepare NetCDF inputs
- **SWOT SSH** NetCDF (gridded product for your region/time). Variable typically: *sea surface height* (name varies by product).  
- **SST** NetCDF on similar dates (L2/L3/Blended).  
- **CHL** NetCDF (MODIS L3 chlor_a; PACE/OCI products when gridded).

> If lon/lat names or units differ, use the script’s CLI flags to specify the variable names.

### 4.3 Build the NPZ bundle
```bash
python /mnt/data/build_env_npz_from_netcdf.py \
  --ssh swot_ssh.nc --sst sst.nc --chl chl.nc \
  --lon-var lon --lat-var lat --ssh-var ssh --sst-var sst --chl-var chlor_a \
  --out /path/to/env_fields_v2_from_realdata.npz
```
- The script:
  1. Loads all NetCDFs with **xarray**.
  2. Chooses the **SSH** grid as reference; interpolates **SST** and **CHL** to that grid.
  3. Computes **U, V** (geostrophy), **EKE**, **|∇SST|**, **Okubo–Weiss**, **eddy cores/edges**,
     **distances to cores/edges**, and synthetic **bathymetry/MLD** placeholders (replace later
     with your bathy/MLD if available).
  4. Writes a complete **NPZ** with the keys listed above.

> Replace synthetic **BATHY/MLD** in the script with your preferred sources (e.g., GEBCO/ETOPO1, Argo MLD).

---

## 5) Tag file schema

Minimum columns:
```
time (ISO8601), lon (deg), lat (deg)
```
Optional:
```
depth_m
```
Tips:
- Sort by time and **resample** to a uniform cadence (e.g., 6h) for better numerical stability.
- The Streamlit app does this automatically for visualization and fitting.

---

## 6) Notes on forecasting & features

- **Forecast model** = advection by geostrophic **U, V** + bias **up the gradient** of the fitted **HSI**; adds velocity persistence and small noise for realism.
- **HSI drivers** include: eddy polarity (SSH sign), **EKE**, **SST fronts** (|∇SST|), distances to **eddy core/edge**, **bathymetry**, distance to **shelf break** (~200 m), and **MLD**/thermocline proxy.
- **Eddy cores** detected via **Okubo–Weiss** (< 5th percentile of W); **edges** by top 20% of |∇SSH|. Tune these thresholds as needed.
- **Safety**: aggregate or blur outputs as appropriate to protect sensitive species locations.

---

## 7) Troubleshooting

- **Bad or empty eddy masks**: check SSH units and gradients; adjust thresholds.
- **Weird forecast drift**: verify lon/lat ordering, units (meters vs degrees), and `f` sign (Coriolis).
- **Interpolation issues**: ensure SST/CHL are on realistic ranges after interpolation.
- **Performance**: reduce grid size or ensemble size, or precompute distances once per day.

---

## 8) Credits & Data Sources (official portals)

- **PACE mission** (hyperspectral ocean color; public data access began April 11, 2024).  
- **MODIS‑Aqua** (long‑term chlorophyll/SST from OB.DAAC L2/L3).  
- **SWOT** (high‑resolution SSH from PO.DAAC).

> Follow the usage policies of the respective data centers when using non‑U.S. Government sites.

---

## 9) File paths in this package

- Notebooks: `/mnt/data/Shark_Habitat_Model_Notebook.ipynb`, `/mnt/data/Shark_Habitat_Model_v2.ipynb`
- Apps: `/mnt/data/app_streamlit.py`, `/mnt/data/bokeh_app/main.py`
- Demo data: `/mnt/data/env_fields_v2.npz`, `/mnt/data/example_tag_data.csv`
- Utility: `/mnt/data/build_env_npz_from_netcdf.py`

Happy modeling!
