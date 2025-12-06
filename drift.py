import numpy as np 
import pandas as pd 
import xarray as xr


R_EARTH=6371000.0 # radii in meters
VG=0/4 #glider speed
def segment_length_m(lonA,latA,lonB,latB):
    latAr,latBr=np.deg2rad(latA),np.deg2rad(latA)
    dlat=latBr-latAr
    dlon=np.deg2rad(lonB-lonA)
    lat_mean=0.5*(latAr+latBr)

    dy=dlat*R_EARTH
    dx=dlon*R_EARTH*np.cos(lat_mean)
    return np.hypot(dx,dy)
def estimate_segment_drift(lonA,latA,lonB,latB,
                           u_par_field,u_perp_field,
                           VG):
    """
    Estimate time and cross-track drift for a single segment
    from (lonA, latA) to (lonB, latB).

    u_par_field, u_perp_field are xarray DataArrays on (lat, lon)
    with along-track and cross-track components relative to the track.
    VG is glider through-water speed (m/s).
    """
    # segment length
    L=segment_length_m(lonA,latA,lonB,latB)
    if L==0:
        return np.nan,np.nan,np.nan # T_hours, drift_km, R
    # midpoint
    lon_mid=0.5*(lonA+lonB)
    lat_mid=0.5*(latA+latB)
    ## Interprolating currents at the above midpoints
    u_par_mid=float(u_par_field.interp(lon=lon_mid,lat=lat_mid))
    u_perp_mid=-float(u_perp_field.interp(lon=lon_mid,lat=lat_mid))
    # u_perp_mid has a negative float, early test versions of advective drift had signs flipped, easiest fix

    # Along track speed
    V_eff=VG+u_par_mid
    if V_eff<=0.02: #flag if crazy slow
        return np.nan,np.nan,np.inf
    
    # Drift over that same distance
    T=L/V_eff # time in seconds=length/speed travelled 
    drift_m=u_par_mid*T
    drift_km=drift_m/1000 
    T_hours=T/3600
    R=abs(u_perp_mid)/max(VG,1e-6) # this is a ratio of drift severity, kind of shaky on this, but it checks out so far
    return T_hours, drift_km, R


### from the files, should fix the empty sheets
def compute_mission_drift(mission_csv_path, nc_path, vg):
    print(f"[DRIFT] Reading mission CSV from: {mission_csv_path}")
    mission_df = pd.read_csv(mission_csv_path)
    print(f"[DRIFT] mission_df rows: {len(mission_df)}")

    print(f"[DRIFT] Reading NC from: {nc_path}")
    ds = xr.open_dataset(nc_path)

    # You’ll adapt this part to your actual column names/variables
    drift_rows = []

    for i in range(len(mission_df) - 1):
        lonA = mission_df.loc[i, "lon"]
        latA = mission_df.loc[i, "lat"]
        lonB = mission_df.loc[i+1, "lon"]
        latB = mission_df.loc[i+1, "lat"]

        # sample the along/cross fields at the segment location
        #taking variable name from nc file here, renaming the variables in m_p
        u_par_field = ds["track_along"]  # adjust names
        u_perp_field = ds["track_across"]

        hours, drift_km, cumul_drift_km, R = estimate_segment_drift(
            lonA, latA, lonB, latB,
            u_par_field, u_perp_field,
            VG
        )

        drift_rows.append([hours, drift_km, cumul_drift_km, R])

    drift_df = pd.DataFrame(
        drift_rows,
        columns=["hours", "drift_km", "cumul_drift_km", "R"]
    )
    print(f"[DRIFT] drift_df rows: {len(drift_df)}")
    return drift_df


# Optional: keep a small debug block for standalone testing
""" if __name__ == "__main__":
    mission_csv = "path/to/test_mission_summary.csv"
    nc_path = "path/to/test_upar_uperp.nc"
    VG = 0.25
    df = compute_drift_for_mission(mission_csv, nc_path, VG)
    df.to_csv("test_drift_summary.csv", index=False) """