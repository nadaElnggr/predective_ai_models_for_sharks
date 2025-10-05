# app_streamlit.py
# ──────────────────────────────────────────────────────────────────────────────
# Shark Movement Forecast — species-aware (uses all_tag_data_combined.csv)
# - Sidebar species selector (from combined file) OR optional CSV upload
# - Robust guards to avoid crashes on short/invalid tracks
# - More defensive env loading (synthetic_env_fields.npz → env_fields_v2.npz)
# - Basic logistic step-selection fit + HSI-based drifted advection forecast
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Shark Movement Forecast (NASA fields)", layout="wide")
st.title("Shark Movement Forecast — eddies • fronts • shelf • MLD")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_env(npz_path):
    env = np.load(npz_path)
    data = {k: env[k] for k in env.files}
    return data

def env_bounds(env):
    """
    Get lon/lat bounds from env; fall back to min/max of coordinate arrays if constants not present.
    """
    if "LON_MIN" in env and "LON_MAX" in env and "LAT_MIN" in env and "LAT_MAX" in env:
        lon_min = float(env["LON_MIN"])
        lon_max = float(env["LON_MAX"])
        lat_min = float(env["LAT_MIN"])
        lat_max = float(env["LAT_MAX"])
    else:
        lons = np.asarray(env.get("lons"))
        lats = np.asarray(env.get("lats"))
        if lons is None or lats is None or lons.size < 2 or lats.size < 2:
            st.error("Environment file is missing coordinate grids ('lons', 'lats') or they are too small.")
            st.stop()
        lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
        lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    return lon_min, lon_max, lat_min, lat_max

def get_grid(env):
    lons = np.asarray(env.get("lons"))
    lats = np.asarray(env.get("lats"))
    if lons is None or lats is None:
        st.error("Environment file must contain 'lons' and 'lats' arrays.")
        st.stop()
    return lons, lats

# ──────────────────────────────────────────────────────────────────────────────
# Interpolation & feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _find_index(arr, val):
    """Return the left index for val in a sorted array, clipped to valid range-2."""
    i = np.searchsorted(arr, val) - 1
    return int(np.clip(i, 0, len(arr) - 2))

def bilinear(field2d, lon, lat, lons, lats):
    """
    Bilinear interpolation of a field on [lats, lons] grid.
    Expects field2d shape = (Ny, Nx) with lats increasing and lons increasing.
    """
    field2d = np.asarray(field2d)
    ix = _find_index(lons, lon)
    iy = _find_index(lats, lat)
    x0, x1 = lons[ix], lons[ix+1]
    y0, y1 = lats[iy], lats[iy+1]

    # Avoid divide-by-zero if degenerate
    dx = (x1 - x0) if (x1 - x0) != 0 else 1e-12
    dy = (y1 - y0) if (y1 - y0) != 0 else 1e-12
    wx = (lon - x0) / dx
    wy = (lat - y0) / dy

    f00 = field2d[iy, ix]
    f10 = field2d[iy, ix+1]
    f01 = field2d[iy+1, ix]
    f11 = field2d[iy+1, ix+1]
    return (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11

def extract_features_at(lon_pts, lat_pts, env):
    """
    Extract environmental features at provided lon/lat arrays.
    Returns standardized features X, and (mu, sigma) for standardization.
    """
    lons, lats = get_grid(env)

    # Layers: only add if present in env
    layers = []
    names  = []

    for key in ["SST", "CHL", "SSH", "EKE", "SST_FRONT", "BATHY", "MLD"]:
        if key in env:
            layers.append(env[key]); names.append(key)

    # Derived proximity to structures (if distances present)
    # (higher = nearer after exp-decay transform)
    def prox_from_dist(dist_m, L=80000.0):
        d = max(float(dist_m), 0.0)
        return math.exp(-d / L)

    has_core  = "DIST_TO_CORE_M"  in env
    has_edge  = "DIST_TO_EDGE_M"  in env
    has_shelf = "DIST_TO_SHELF_M" in env

    rows = []
    for lo, la in zip(lon_pts, lat_pts):
        feats = []
        for arr in layers:
            feats.append(bilinear(arr, lo, la, lons, lats))
        # add proximities if available
        if has_core:
            d_core = bilinear(env["DIST_TO_CORE_M"], lo, la, lons, lats)
            feats.append(prox_from_dist(d_core, L=60000.0))
            names_core = "PROX_CORE"
        if has_edge:
            d_edge = bilinear(env["DIST_TO_EDGE_M"], lo, la, lons, lats)
            feats.append(prox_from_dist(d_edge, L=80000.0))
            names_edge = "PROX_EDGE"
        if has_shelf:
            d_shelf = bilinear(env["DIST_TO_SHELF_M"], lo, la, lons, lats)
            feats.append(prox_from_dist(d_shelf, L=120000.0))
            names_shelf = "PROX_SHELF"

        rows.append(feats)

    X = np.asarray(rows, dtype=float)

    # Standardize features (avoid zero std)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig[sig < 1e-8] = 1.0
    Xn = (X - mu) / sig
    return Xn, mu, sig

# ──────────────────────────────────────────────────────────────────────────────
# Simple logistic regression (L2) for step selection
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def fit_logit_newton(X, y, lam=2e-2, max_iter=100):
    """
    Fit logistic regression with L2 penalty via Newton-Raphson.
    X must already include a column of ones for intercept.
    """
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        z = X @ w
        p_hat = sigmoid(z)
        W = p_hat * (1.0 - p_hat)
        # add ridge to Hessian
        H = (X.T * W) @ X + lam * np.eye(p)
        g = X.T @ (p_hat - y) + lam * w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if np.linalg.norm(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new
    return w

# ──────────────────────────────────────────────────────────────────────────────
# HSI & forecasting
# ──────────────────────────────────────────────────────────────────────────────

def hsi_prob(lon, lat, coef, mu, sig, env):
    """
    Compute HSI probability at a single lon/lat by extracting features on the fly.
    """
    X1, _, _ = extract_features_at([lon], [lat], env)        # standardized features
    X1 = np.hstack([np.ones((X1.shape[0], 1)), X1])          # add intercept
    p = sigmoid(X1 @ coef)[0]
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))

def predict(lon0, lat0, coef, mu, sig, env, N_fore=24, dt_hours=6.0, N_ens=30):
    """
    Drifted advection: U/V if available; add small bias toward HSI gradient.
    """
    lons, lats = get_grid(env)
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = env_bounds(env)

    # grid spacing to meters (approx)
    dx_m = 111e3 * math.cos(math.radians(lat0))
    dy_m = 111e3

    def one_traj(lon_init, lat_init, seed):
        rng = np.random.default_rng(seed)
        lon, lat = float(lon_init), float(lat_init)
        traj = [(lon, lat)]
        vx_prev = vy_prev = 0.0

        for _ in range(int(N_fore)):
            # base drift from currents (if available)
            if "U" in env and "V" in env:
                u = bilinear(env["U"], lon, lat, lons, lats)   # m/s
                v = bilinear(env["V"], lon, lat, lons, lats)   # m/s
            else:
                u = v = 0.0

            # bias toward higher HSI via finite-difference gradient
            eps = 0.02  # deg
            h_e = hsi_prob(min(lon+eps, LON_MAX), lat, coef, mu, sig, env)
            h_w = hsi_prob(max(lon-eps, LON_MIN), lat, coef, mu, sig, env)
            h_n = hsi_prob(lon, min(lat+eps, LAT_MAX), coef, mu, sig, env)
            h_s = hsi_prob(lon, max(lat-eps, LAT_MIN), coef, mu, sig, env)
            dhdx = (h_e - h_w) / (2 * eps * dx_m)
            dhdy = (h_n - h_s) / (2 * eps * dy_m)
            bias_u = 500.0 * dhdx
            bias_v = 500.0 * dhdy

            # momentum + a touch of noise
            vx = 0.6 * (u + bias_u) + 0.3 * vx_prev + 0.05 * rng.normal()
            vy = 0.6 * (v + bias_v) + 0.3 * vy_prev + 0.05 * rng.normal()

            # advance in degrees
            dt = dt_hours * 3600.0
            lon += (vx * dt) / dx_m
            lat += (vy * dt) / dy_m

            # clamp to domain
            lon = float(np.clip(lon, LON_MIN, LON_MAX))
            lat = float(np.clip(lat, LAT_MIN, LAT_MAX))

            vx_prev, vy_prev = vx, vy
            traj.append((lon, lat))

        return np.array(traj)

    seeds = np.arange(N_ens)
    return [one_traj(lon0, lat0, int(s)) for s in seeds]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls — env & tags (species-aware)
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Inputs")

# ENV loader
env_file = st.sidebar.file_uploader("Environmental fields (.npz)", type=["npz"])
if env_file is None:
    st.sidebar.info("Using packaged fields (auto-fallback).")
    try:
        env = load_env("synthetic_env_fields.npz")
    except Exception:
        env = load_env("env_fields_v2.npz")
else:
    env = load_env(env_file)

# TAG data — optional upload; default: combined file with species selector
tags_file = st.sidebar.file_uploader(
    "Optional: upload a tag CSV (columns: time,lon,lat[,depth_m])",
    type=["csv"]
)

if tags_file is None:
    # Use combined dataset
    try:
        df_all = pd.read_csv("all_tag_data_combined.csv", parse_dates=["time"])
    except FileNotFoundError:
        st.error("Missing all_tag_data_combined.csv in the app directory.")
        st.stop()

    if "species" not in df_all.columns:
        st.error("Combined dataset is missing a 'species' column.")
        st.stop()

    # Select species
    species_list = sorted(df_all["species"].dropna().unique().tolist())
    if not species_list:
        st.error("No species found in combined dataset.")
        st.stop()
    selected_species = st.sidebar.selectbox("Species", species_list, index=0)

    tag_df = (
        df_all[df_all["species"] == selected_species]
        .loc[:, ["time", "lon", "lat", "depth_m"]]
        .sort_values("time")
        .reset_index(drop=True)
    )
    st.sidebar.success(f"Using {len(tag_df)} points for species: {selected_species}")
else:
    # Uploaded CSV path
    up = pd.read_csv(tags_file, parse_dates=["time"], infer_datetime_format=True)
    for col in ["lon", "lat"]:
        if col not in up.columns:
            st.error(f"Uploaded CSV is missing required column: '{col}'.")
            st.stop()
    if "depth_m" not in up.columns:
        up["depth_m"] = pd.NA
    tag_df = (
        up.loc[:, ["time", "lon", "lat", "depth_m"]]
        .sort_values("time")
        .reset_index(drop=True)
    )
    st.sidebar.success(f"Using uploaded tag with {len(tag_df)} points.")

# Clean/guard
tag_df = tag_df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
if tag_df.empty:
    st.error("No valid lon/lat rows found after cleaning. Select a different species or upload a valid CSV.")
    st.stop()

# Require a minimum number of points to fit the model
MIN_POINTS = 10
if len(tag_df) < MIN_POINTS:
    st.error(f"Not enough tag positions for modelling (got {len(tag_df)}, need ≥ {MIN_POINTS}). Try another species or upload a richer CSV.")
    st.stop()

# Region & forecast controls
st.sidebar.header("Region & forecast")
lon_min, lon_max, lat_min, lat_max = env_bounds(env)

lon_rng = st.sidebar.slider(
    "Longitude range", float(lon_min), float(lon_max), (float(lon_min), float(lon_max))
)
lat_rng = st.sidebar.slider(
    "Latitude range", float(lat_min), float(lat_max), (float(lat_min), float(lat_max))
)
N_fore = st.sidebar.number_input("Forecast steps (6 h per step)", min_value=8, max_value=96, value=24, step=4)
N_ens  = st.sidebar.number_input("Ensemble members", min_value=10, max_value=100, value=30, step=5)
dt_hours = st.sidebar.selectbox("Step size (hours)", [3.0, 6.0, 12.0], index=1)

# ──────────────────────────────────────────────────────────────────────────────
# Fit HSI weights (logistic step-selection)
# ──────────────────────────────────────────────────────────────────────────────

# Used steps: consecutive moves; available: random proposals around previous point
used_lon = tag_df["lon"].values[1:]
used_lat = tag_df["lat"].values[1:]
n = used_lon.size
K = 5  # proposals per used step

if n <= 0:
    st.error("Insufficient consecutive points to compute used steps. Try another species or upload a richer CSV.")
    st.stop()

avail_lon, avail_lat = [], []
rng = np.random.default_rng(0)
for i in range(n):
    lo0 = tag_df["lon"].values[i]
    la0 = tag_df["lat"].values[i]
    props_lo = lo0 + 0.6 * np.sqrt(float(dt_hours)/6.0) * rng.normal(size=K)
    props_la = la0 + 0.6 * np.sqrt(float(dt_hours)/6.0) * rng.normal(size=K)
    props_lo = np.clip(props_lo, float(lon_min), float(lon_max))
    props_la = np.clip(props_la, float(lat_min), float(lat_max))
    avail_lon.append(props_lo)
    avail_lat.append(props_la)

avail_lon = np.array(avail_lon).reshape(-1)
avail_lat = np.array(avail_lat).reshape(-1)

if used_lon.size == 0 or used_lat.size == 0:
    st.error("Insufficient used steps (after cleaning). Try another species or upload a richer CSV.")
    st.stop()
if avail_lon.size == 0 or avail_lat.size == 0:
    st.error("Failed to generate available steps. Increase track length or adjust dt_hours.")
    st.stop()

# Features for used and available steps
X_used, mu, sig = extract_features_at(used_lon, used_lat, env)
X_av,   _,  _   = extract_features_at(avail_lon, avail_lat, env)

# Build design matrix with intercept
X = np.vstack([X_used, X_av])
y = np.hstack([np.ones(X_used.shape[0]), np.zeros(X_av.shape[0])])
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Fit logistic regression
coef = fit_logit_newton(X, y, lam=2e-2, max_iter=100)
st.success("HSI weights fitted from current tag.")

# ──────────────────────────────────────────────────────────────────────────────
# Run forecast & plot
# ──────────────────────────────────────────────────────────────────────────────

if st.button("Run forecast"):
    last_lon = float(tag_df["lon"].values[-1])
    last_lat = float(tag_df["lat"].values[-1])
    trajs = predict(last_lon, last_lat, coef, mu, sig, env,
                    N_fore=int(N_fore), dt_hours=float(dt_hours), N_ens=int(N_ens))

    # Layer selection (only those available)
    available_layers = [k for k in ["SSH","SST","CHL","EKE","SST_FRONT","BATHY","MLD"] if k in env]
    available_layers.append("HSI (computed)")
    layer_name = st.selectbox("Map layer", available_layers, index=0)

    # Build layer to show
    if layer_name == "HSI (computed)":
        # compute HSI on grid (may be heavy on large grids)
        lons, lats = get_grid(env)
        H = np.zeros((len(lats), len(lons)), dtype=float)
        for j, la in enumerate(lats):
            for i, lo in enumerate(lons):
                H[j, i] = hsi_prob(float(lo), float(la), coef, mu, sig, env)
        layer = H
    else:
        layer = env[layer_name]

    # Crop to selected region
    lons, lats = get_grid(env)
    i0 = int(np.searchsorted(lons, lon_rng[0]))
    i1 = int(np.searchsorted(lons, lon_rng[1]))
    j0 = int(np.searchsorted(lats, lat_rng[0]))
    j1 = int(np.searchsorted(lats, lat_rng[1]))
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, len(lons)-1), min(j1, len(lats)-1)
    if i1 <= i0: i0, i1 = 0, len(lons)-1
    if j1 <= j0: j0, j1 = 0, len(lats)-1

    # Plot
    fig = plt.figure(figsize=(7, 5))
    plt.title(f"{layer_name}")
    plt.imshow(layer[j0:j1, i0:i1], origin="lower",
               extent=[lons[i0], lons[i1], lats[j0], lats[j1]])
    tail = tag_df.tail(16)
    plt.plot(tail["lon"].values, tail["lat"].values, linewidth=2)
    for tr in trajs:
        plt.plot(tr[:, 0], tr[:, 1], linestyle="--", linewidth=0.8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Click **Run forecast** to generate ensemble trajectories.")
