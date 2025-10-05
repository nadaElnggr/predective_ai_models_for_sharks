#!/usr/bin/env python3
"""
Build an environment NPZ bundle (env_fields_v2.npz) from real NetCDF inputs.

This script expects:
  - SSH from SWOT (grid reference)
  - SST (any gridded product)
  - CHL (MODIS/PACE chlorophyll)

It interpolates SST/CHL to the SSH grid, derives U/V geostrophic currents,
EKE, SST fronts, Okubo–Weiss, eddy masks, and distances. It also creates a
simple synthetic bathymetry/MLD unless you provide alternatives.

Example:
  python build_env_npz_from_netcdf.py \
    --ssh swot_ssh.nc --sst sst.nc --chl chl.nc \
    --lon-var lon --lat-var lat --ssh-var ssh --sst-var sst --chl-var chlor_a \
    --out env_fields_v2_from_realdata.npz

Dependencies: numpy, xarray, netCDF4
"""
import argparse
import numpy as np
import xarray as xr

def compute_dx_dy_m(lat_deg):
    lat0 = np.deg2rad(np.asarray(lat_deg).mean())
    dx_m = 111000.0 * np.cos(lat0)
    dy_m = 111000.0
    return float(dx_m), float(dy_m), float(2 * 7.2921e-5 * np.sin(lat0)), 9.81

def grad(field, dy_m, dx_m):
    dY, dX = np.gradient(field, dy_m, dx_m)
    return dY, dX

def okubo_weiss(u, v, dy_m, dx_m):
    dU_dy, dU_dx = np.gradient(u, dy_m, dx_m)
    dV_dy, dV_dx = np.gradient(v, dy_m, dx_m)
    zeta = dV_dx - dU_dy
    s_n = dU_dx - dV_dy
    s_s = dV_dx + dU_dy
    W = s_n**2 + s_s**2 - zeta**2
    return W

def distance_to_mask(mask, dx_m, dy_m):
    # brute-force but robust for moderate grids
    ny, nx = mask.shape
    Y, X = np.indices((ny, nx))
    pts = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    if pts.size == 0:
        return np.full(mask.shape, np.nan)
    # build coords
    all_pts = np.column_stack([X.ravel(), Y.ravel()])
    out = np.empty(all_pts.shape[0], dtype=float)
    step = 4000
    for i in range(0, out.size, step):
        chunk = all_pts[i:i+step]
        d2 = ((chunk[:,None,:] - pts[None,:,:])**2).sum(axis=2).min(axis=1)
        out[i:i+step] = np.sqrt(d2)
    # convert pixel distances to meters
    # approximate scaling: dx_m per x-step, dy_m per y-step
    # isotropic approximation using average meter/px
    m_per_px = 0.5*(dx_m + dy_m)
    return (out.reshape((ny, nx)) * m_per_px)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", required=True, help="SWOT SSH NetCDF")
    ap.add_argument("--sst", required=True, help="SST NetCDF")
    ap.add_argument("--chl", required=True, help="Chlorophyll NetCDF")
    ap.add_argument("--lon-var", default="lon", help="Longitude variable name")
    ap.add_argument("--lat-var", default="lat", help="Latitude variable name")
    ap.add_argument("--ssh-var", default="ssh", help="SSH variable name")
    ap.add_argument("--sst-var", default="sst", help="SST variable name")
    ap.add_argument("--chl-var", default="chlor_a", help="Chlorophyll variable name")
    ap.add_argument("--out", required=True, help="Output NPZ path")
    args = ap.parse_args()

    ds_ssh = xr.open_dataset(args.ssh)
    ds_sst = xr.open_dataset(args.sst)
    ds_chl = xr.open_dataset(args.chl)

    lons = ds_ssh[args.lon_var].values
    lats = ds_ssh[args.lat_var].values
    if lons.ndim == 1 and lats.ndim == 1:
        # expected 1D coords
        pass
    else:
        raise ValueError("This script expects 1D lon/lat. Regrid beforehand if your product is swath-like.")

    SSH = np.asarray(ds_ssh[args.ssh_var]).astype(float)
    # Choose one time slice if needed
    if SSH.ndim == 3:
        SSH = SSH[0, :, :]

    # Interpolate SST & CHL to SSH grid
    SST = np.asarray(ds_sst[args.sst_var].interp({args.lat_var: lats, args.lon_var: lons}).load()).astype(float)
    if SST.ndim == 3: SST = SST[0, :, :]
    CHL = np.asarray(ds_chl[args.chl_var].interp({args.lat_var: lats, args.lon_var: lons}).load()).astype(float)
    if CHL.ndim == 3: CHL = CHL[0, :, :]

    dx_m, dy_m, f, g = compute_dx_dy_m(lats)

    dSSH_dy, dSSH_dx = grad(SSH, dy_m, dx_m)
    U = -(g / f) * dSSH_dy
    V =  (g / f) * dSSH_dx
    SPEED = np.hypot(U, V)

    dSST_dy, dSST_dx = grad(SST, dy_m, dx_m)
    SST_FRONT = np.hypot(dSST_dx, dSST_dy)

    U_an = U - np.nanmean(U); V_an = V - np.nanmean(V)
    EKE = 0.5*(U_an**2 + V_an**2)

    W = okubo_weiss(U, V, dy_m, dx_m)
    core_thresh = np.nanpercentile(W, 5.0)
    EDDY_CORE_MASK = (W < core_thresh)

    SSH_grad = np.hypot(dSSH_dx, dSSH_dy)
    edge_thresh = np.nanpercentile(SSH_grad, 80.0)
    EDDY_EDGE_MASK = (SSH_grad >= edge_thresh)

    # Eddy polarity mask
    WARM_MASK = (SSH > 0.0).astype(np.int8)

    # Simple synthetic bathymetry & MLD (replace with your datasets if available)
    LON, LAT = np.meshgrid(lons, lats)
    xi = (LON - LON.min()) / (LON.max() - LON.min())
    BATHY = 50.0 + 5000.0 * xi**2  # 50 m shelf -> ~5 km abyssal
    SHELF_MASK = (np.abs(BATHY - 200.0) <= 25.0).astype(np.int8)
    # Distance rasters
    DIST_TO_CORE_M  = distance_to_mask(EDDY_CORE_MASK, dx_m, dy_m)
    DIST_TO_EDGE_M  = distance_to_mask(EDDY_EDGE_MASK, dx_m, dy_m)
    DIST_TO_SHELF_M = distance_to_mask(SHELF_MASK.astype(bool), dx_m, dy_m)

    # MLD proxy: shallower in warm eddies, deeper in cold
    SST_norm = (SST - np.nanmin(SST)) / (np.nanmax(SST) - np.nanmin(SST) + 1e-12)
    MLD = 20.0 + 60.0*(1.0 - SST_norm) - 10.0*(WARM_MASK.astype(bool))

    np.savez_compressed(
        args.out,
        lons=lons, lats=lats,
        SSH=SSH, U=U, V=V, SPEED=SPEED,
        SST=SST, CHL=CHL,
        W=W, EDDY_CORE_MASK=EDDY_CORE_MASK.astype(np.int8), EDDY_EDGE_MASK=EDDY_EDGE_MASK.astype(np.int8),
        WARM_MASK=WARM_MASK,
        EKE=EKE, SST_FRONT=SST_FRONT,
        BATHY=BATHY, SHELF_MASK=SHELF_MASK,
        MLD=MLD,
        DIST_TO_CORE_M=DIST_TO_CORE_M, DIST_TO_EDGE_M=DIST_TO_EDGE_M, DIST_TO_SHELF_M=DIST_TO_SHELF_M,
        dx_m=dx_m, dy_m=dy_m, f=f, g=g,
        LON_MIN=float(lons.min()), LON_MAX=float(lons.max()),
        LAT_MIN=float(lats.min()), LAT_MAX=float(lats.max())
    )
    print("Wrote bundle:", args.out)

if __name__ == "__main__":
    main()
