import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt 
# ---- CONFIG ----
NC_FILE = "/home/ps1532/ggs3/GGS3/products/2025_11_15/data/test_mission_2025111500_ESPC_dac.nc"  
VG = 0.40     # adjust to reflect glider speed                

# ---- LOAD DATA FROM NETCDF ----
ds = xr.open_dataset(NC_FILE)

# Depth-averaged current components relative to track
u_par = ds["track_along"]      # (lat, lon), m/s
u_perp = ds["track_cross"]     # (lat, lon), m/s
magnitude = ds["magnitude"]

# Grid coordinates these fields live on
grid_lon = ds["lon"]           # or ds["longitude"]
grid_lat = ds["lat"]           # or ds["latitude"]

#
test_lon=-68
test_lat=37 
v_raw=float(ds["v"].interp(lon=test_lon,lat=test_lat))
u_perp_here=-float(u_perp.interp(lon=test_lon,lat=test_lat))
print(f"\nSign check at ({test_lon:.2f}, {test_lat:.2f})")
print(f"  depth-avg v_raw (north+ / south-): {v_raw:.3f} m/s")
print(f"  track_cross u_perp:                {u_perp_here:.3f} m/s\n")

# CSV read
Mission_path = "/home/ps1532/ggs3/GGS3/products/2025_08_08_pathfinding_output/data/test_mission_20250802_ESPC_mission_path.csv"
path_df=pd.read_csv(Mission_path)
path_lon=path_df["lon"].to_numpy()
path_lat=path_df["lat"].to_numpy()

# ---- HELPER: METRIC CONVERSION ----
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

# ---- PHASE 1 DRIFT ANALYSIS ----
records = []

for i in range(len(path_lon) - 1):
    lonA, latA = float(path_lon[i]),     float(path_lat[i])
    lonB, latB = float(path_lon[i + 1]), float(path_lat[i + 1])

    # Segment length
    L = segment_length_m(lonA, latA, lonB, latB)  # meters
    if L == 0:
        continue

    # Midpoint for sampling currents
    lon_mid = 0.5 * (lonA + lonB)
    lat_mid = 0.5 * (latA + latB)

    # Interpolate along/cross currents at the midpoint
    u_par_mid = float(u_par.interp(lon=lon_mid, lat=lat_mid))
    u_perp_mid = -float(u_perp.interp(lon=lon_mid, lat=lat_mid))

    # Effective along-track speed and time on leg
    V_eff = VG + u_par_mid
    # Avoid weirdness if V_eff is very small or negative
    if V_eff <= 0.02:  # 2 cm/s floor, you can adjust
        T = np.nan
        drift_km = np.nan
    else:
        T = L / V_eff     # seconds
        drift_m = u_perp_mid * T
        drift_km = drift_m / 1000.0

    T_hours = T / 3600.0 if np.isfinite(T) else np.nan
    R = abs(u_perp_mid) / max(VG, 1e-6)

    records.append(
        {
            "seg_index": i,
            "lon_mid": lon_mid,
            "lat_mid": lat_mid,
            "L_km": L / 1000.0,
            "u_parallel_mps": u_par_mid,
            "u_perp_mps": u_perp_mid,
            "V_eff_mps": V_eff,
            "T_hours": T_hours,
            "drift_km": drift_km,
            "R": R,
        }
    )

df = pd.DataFrame.from_records(records)

print(df.head())
print()
print("Top 10 segments by |drift_km|:")
print(df.reindex(df["drift_km"].abs().sort_values(ascending=False).index).head(10))

# Optionally save to CSV for inspection
# df.to_csv("phase1_drift_summary.csv", index=False)
# print("\nSaved summary to phase1_drift_summary.csv")
df = pd.DataFrame.from_records(records)
# ---- DEBUG: look at the meander window ----
mask = (df["lon_mid"] > -66.5) & (df["lon_mid"] < -65.0)
df_slice = df[mask].sort_values("lon_mid")

print("\nMeander slice:")
print(df_slice[["lon_mid","u_perp_mps","drift_km","R"]])

# ---- rest of your summaries / plots ----
print("\nFirst few segments:")
print(df.head())
# ---- BASIC SUMMARY ----
print("\nFirst few segments:")
print(df.head())

print("\nDrift / ratio summary:")
print(df[["drift_km", "R", "T_hours"]].describe())

# Sort by |drift_km| to see the worst segments
df_sorted = df.reindex(df["drift_km"].abs().sort_values(ascending=False).index)

print("\nTop 10 segments by |drift_km|:")
print(df_sorted.head(10)[
    ["seg_index", "L_km", "drift_km", "R", "T_hours", "lon_mid", "lat_mid"]
])

# ---- FLAG SEGMENTS BY SEVERITY (TWEAK THRESHOLDS AS YOU LIKE) ----
df["flag"] = "ok"
df.loc[(df["drift_km"].abs() > 20) | (df["R"] > 0.7), "flag"] = "warn"
df.loc[(df["drift_km"].abs() > 50) | (df["R"] > 1.0), "flag"] = "bad"

print("\nFlag counts:")
print(df["flag"].value_counts())

print("\nFlagged segments (warn/bad):")
print(df[df["flag"] != "ok"][
    ["seg_index", "L_km", "drift_km", "R", "T_hours", "lon_mid", "lat_mid", "flag"]
])

# ---- SAVE CSV ----
df.to_csv("phase1_drift_summary.csv", index=False)
print("\nSaved summary to phase1_drift_summary.csv")

# quick visualizer 
plt.figure(figsize=(8,4))
sc = plt.scatter(df["lon_mid"], df["lat_mid"],
                 c=df["u_perp_mps"],
                 s=15,
                 cmap="coolwarm",
                 vmin=-0.5, vmax=0.5)  # clamp to highlight sign
plt.colorbar(sc, label="cross-track current u_perp (m/s)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Raw cross-track current along test path\n(red = north of track, blue = south)")
plt.tight_layout()
plt.savefig("phase1_u_perp_map.png", dpi=150)

# drift map

print("\nDrift stats:")
print(df["drift_km"].describe())
print(df["drift_km"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]))

# pick symmetric vmin/vmax from the 5–95% range
lo, hi = df["drift_km"].quantile([0.05, 0.95])
vmax = max(abs(lo), abs(hi))
vmin = -vmax
print(f"\nUsing vmin={vmin:.1f}, vmax={vmax:.1f} for drift colormap")


plt.figure(figsize=(8,4))
plt.pcolormesh(grid_lon,grid_lat,magnitude,shading="auto") #,cmap="cmocean"
plt.colorbar(label="cucrrent speed(m/s)")
sc=plt.scatter(
    df["lon_mid"],
    df["lat_mid"],
    c=df["drift_km"],
    s=5,
    cmap="coolwarm",
    vmin=vmin,
    vmax=vmax,
    edgecolors="none",
    alpha=0.9,
    zorder=5,
)
plt.colorbar(sc,label="drift per leg(km)\n(red=North,blue=South)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Phase 1 cross-track drift per segment\n(red=North of track,blue=South)")
plt.tight_layout()
plt.savefig("phase1_drift_map.png",dpi=150)
print("Saved map to phase1_drift_map.png")

# Drifting along path
L = df["L_km"].to_numpy()          # segment lengths
drift = df["drift_km"].to_numpy()  # signed per-segment drift

s_along = np.concatenate([[0.0], np.cumsum(L)])        # 0 → total distance
N_cum   = np.concatenate([[0.0], np.cumsum(drift)])    # 0 → net offset

print(f"\nTotal path length: {s_along[-1]:.1f} km")
print(f"Final cumulative offset: {N_cum[-1]:.1f} km")

plt.figure(figsize=(8,4))
plt.axhline(0, color="k", linewidth=1, alpha=0.4)
plt.plot(s_along, N_cum, "-", linewidth=2)
plt.xlabel("Distance along path (km)")
plt.ylabel("Cumulative cross-track offset (km)")
plt.title("Cumulative modeled drift along A* path\n(positive = North/left, negative = South/right)")
plt.tight_layout()
plt.savefig("phase1_cumulative_drift.png", dpi=150)
print("Saved cumulative drift plot to phase1_cumulative_drift.png")