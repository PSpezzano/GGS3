import numpy as np
import pandas as pd
import xarray as xr

import numpy as np

R_EARTH = 6371000.0  # meters


def segment_length_m(lon1, lat1, lon2, lat2):
    """Approximate great-circle distance between two close points (meters)."""
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    lat_mean = 0.5 * (lat1r + lat2r)

    dy = dlat * R_EARTH
    dx = dlon * R_EARTH * np.cos(lat_mean)
    return np.hypot(dx, dy)


def estimate_segment_drift(lonA, latA, lonB, latB,
                           u_par_field, u_perp_field,
                           VG):
    """
    Phase_1 drift logic, but using arbitrary u_par_field/u_perp_field
    DataArrays instead of globals.
    """
    # --- segment length ---
    L = segment_length_m(lonA, latA, lonB, latB)  # meters
    if L == 0:
        return np.nan, np.nan, np.nan,np.nan,np.nan  # T_hours, drift_km, R, perp,par

    # --- midpoint for sampling currents ---
    lon_mid = 0.5 * (lonA + lonB)
    lat_mid = 0.5 * (latA + latB)

    # NOTE: adjust 'lon'/'lat' if your coords are named differently
    u_par_mid = float(u_par_field.interp(lon=lon_mid, lat=lat_mid))
    u_perp_mid = -float(u_perp_field.interp(lon=lon_mid, lat=lat_mid))  # same sign flip

    # --- glider speed: match Phase_1 default if None ---
    if VG is None:
        VG = 0.40  # adjust to match mission.json speed 

    # --- effective along-track speed ---
    V_eff = VG + u_par_mid
    if not np.isfinite(V_eff) or V_eff<=0.02:
        return np.nan,np.nan,np.nan,np.nan,np.nan

    # --- drift over that leg ---
    T = L / V_eff            # seconds
    drift_m = u_perp_mid * T
    drift_km = drift_m / 1000.0
    T_hours = T / 3600.0

    R = abs(u_perp_mid)/max(VG, 1e-6)
    
    return T_hours, drift_km,u_perp_mid, u_par_mid,R


def compute_mission_drift(mission_csv_path, nc_path, vg):
    """
    Standalone helper that reads mission_path.csv and the *_dac.nc file,
    then computes per-segment drift using estimate_segment_drift.
    """
    print(f"[DRIFT] Reading mission CSV from: {mission_csv_path}")
    mission_df = pd.read_csv(mission_csv_path)
    print(f"[DRIFT] mission_df rows: {len(mission_df)}")

    print(f"[DRIFT] Reading NC from: {nc_path}")
    ds = xr.open_dataset(nc_path)

    u_par_field = ds["track_along"]
    u_perp_field = ds["track_cross"]   

    drift_rows = []
    cumul_drift_km = 0.0

    for i in range(len(mission_df) - 1):
        lonA = mission_df.loc[i, "lon"]
        latA = mission_df.loc[i, "lat"]
        lonB = mission_df.loc[i+1, "lon"]
        latB = mission_df.loc[i+1, "lat"]

        T_hours, drift_km,u_par_mid,u_perp_mid, R = estimate_segment_drift(
            lonA, latA, lonB, latB,
            u_par_field, u_perp_field,
            vg,
        )

        if np.isfinite(drift_km):
            cumul_drift_km += drift_km

        drift_rows.append([T_hours, drift_km, cumul_drift_km,
                           u_perp_mid,u_par_mid, R])

    drift_df = pd.DataFrame(
        drift_rows,
        columns=["hours", "drift_km", "cumul_drift_km",
                 'u_perp_mps','u_par_mps', "R"],
    )
    print(f"[DRIFT] drift_df rows: {len(drift_df)}")
    print("[DRIFT] compute_mission_drift columns:", drift_df.columns.tolist())
    return drift_df
