from __future__ import annotations
import argparse, os, json, csv
from datetime import datetime, timezone
from pathlib import Path

from ggs3.sim.drift import DriftConfig, simulate_drift


def parse_time(s: str) -> datetime:
    # accepts ...Z or ISO w/ offset
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_geojson(rows, path: Path):
    coords = [[r["lon"], r["lat"]] for r in rows if r.get("ok", True)]
    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "drift_track"},
            "geometry": {"type": "LineString", "coordinates": coords},
        }]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fc, f)


def parse_extent(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Extent must be 'lat_min,lon_min,lat_max,lon_max'.")
    lat_min, lon_min, lat_max, lon_max = (float(p) for p in parts)
    return lat_min, lon_min, lat_max, lon_max


def _select_dataset_path(
    products_root: Path,
    model: str | None,
    dataset_path: str | None,
    extent=None,
) -> Path:
    if dataset_path is not None:
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset not found: {ds_path}")
        return ds_path

    if model:
        model_key = model.strip().upper()
        if model_key not in {"ESPC", "CMEMS"}:
            raise ValueError(f"Unknown model '{model}'. Expected 'ESPC' or 'CMEMS'.")
        pattern = f"*_{model_key}_dac.nc"
    else:
        pattern = "*_dac.nc"

    nc_paths = sorted(products_root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not nc_paths:
        raise FileNotFoundError(f"No {pattern} files found under {products_root}")
    if extent is None:
        return nc_paths[0]

    import xarray as xr
    req = parse_extent(extent) if isinstance(extent, str) else extent
    req_lat_min, req_lon_min, req_lat_max, req_lon_max = req
    for path in nc_paths:
        try:
            with xr.open_dataset(path) as ds:
                lat_vals = ds["lat"].values
                lon_vals = ds["lon"].values
                lat_min = float(lat_vals.min())
                lat_max = float(lat_vals.max())
                lon_min = float(lon_vals.min())
                lon_max = float(lon_vals.max())
                if (
                    lat_min <= req_lat_min <= lat_max
                    and lat_min <= req_lat_max <= lat_max
                    and lon_min <= req_lon_min <= lon_max
                    and lon_min <= req_lon_max <= lon_max
                ):
                    return path
        except Exception:
            continue

    return nc_paths[0]


def run_drift(
    start_time,
    start_lat,
    start_lon,
    hours,
    dt=600,
    depth_m=None,
    outdir=None,
    extent=None,
    dataset_path=None,
    model=None,
    plot_config=None,
):
    import numpy as np
    import xarray as xr
    from ggs2.model_processing import _interp_uv_at

    if isinstance(start_time, str):
        t0 = parse_time(start_time)
    else:
        t0 = start_time

    duration_s = int(hours * 3600)

    if outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = Path("drift_runs") / stamp
    else:
        outdir = Path(outdir)

    products_root = Path(__file__).resolve().parents[1] / "products"
    ds_path = _select_dataset_path(products_root, model, dataset_path, extent=extent)

    ds = xr.open_dataset(ds_path)
    model_label = (model or ds.attrs.get("model_name") or ds.attrs.get("text_name") or ds_path.name).upper()

    def _select_time_depth(ds_in, t_utc, depth_m_in):
        sub = ds_in
        if "time" in sub.dims or "time" in sub.coords:
            sub = sub.interp(time=np.datetime64(t_utc), method="nearest")
        if "depth" in sub.dims:
            depth_val = depth_m_in
            if depth_val is None:
                depth_vals = sub["depth"].values
                depth_val = float(depth_vals[0]) if len(depth_vals) else 0.0
            sub = sub.interp(depth=depth_val, method="nearest")
        return sub

    def _apply_extent(ds_in, extent_in):
        if extent_in is None:
            return ds_in
        if isinstance(extent_in, str):
            extent_in = parse_extent(extent_in)
        lat_min, lon_min, lat_max, lon_max = extent_in
        lat_vals = ds_in["lat"].values
        lon_vals = ds_in["lon"].values
        lat_slice = slice(lat_min, lat_max) if lat_vals[0] <= lat_vals[-1] else slice(lat_max, lat_min)
        lon_slice = slice(lon_min, lon_max) if lon_vals[0] <= lon_vals[-1] else slice(lon_max, lon_min)
        return ds_in.sel(lat=lat_slice, lon=lon_slice)

    def sample_uv(t_utc, lat, lon, depth_m_in):
        sub = _select_time_depth(ds, t_utc, depth_m_in)
        return _interp_uv_at(sub, lat, lon)

    def _select_uv_fields(ds_in, t_utc, depth_m_in):
        sub = _select_time_depth(ds_in, t_utc, depth_m_in)
        sub = _apply_extent(sub, extent)
        if "u" not in sub or "v" not in sub:
            raise KeyError("Dataset missing u/v variables for streamplot.")
        return sub

    cfg = DriftConfig(
        start_time=t0,
        start_lat=start_lat,
        start_lon=start_lon,
        duration_s=duration_s,
        dt_s=dt,
        depth_m=depth_m,
        stop_on_nan=True,
    )

    rows = simulate_drift(cfg, sample_uv)
    write_csv(rows, outdir / "drift.csv")
    write_geojson(rows, outdir / "drift.geojson")

    # Streamplot over current magnitude with drift track overlay.
    try:
        import matplotlib.pyplot as plt
        import cmocean.cm as cmo
        import cartopy.crs as ccrs
        import cool_maps.plot as cplt
    except Exception as exc:
        raise ImportError(
            "matplotlib, cmocean, cartopy, and cool_maps are required to save the drift streamplot."
        ) from exc

    sub = _select_uv_fields(ds, t0, depth_m)
    data_time = t0.strftime("%Y-%m-%d %H:%M")
    lon = sub["lon"].values
    lat = sub["lat"].values
    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("Need at least 2 lon/lat points for streamplot.")

    lon_u = np.linspace(float(np.min(lon)), float(np.max(lon)), len(lon))
    lat_u = np.linspace(float(np.min(lat)), float(np.max(lat)), len(lat))
    sub_u = sub.interp(lon=lon_u, lat=lat_u, method="linear")
    u = sub_u["u"].values
    v = sub_u["v"].values
    speed = np.hypot(u, v)

    step_lon = max(1, int(len(lon_u) // 200))
    step_lat = max(1, int(len(lat_u) // 200))
    lon_s = lon_u[::step_lon]
    lat_s = lat_u[::step_lat]
    u_s = u[::step_lat, ::step_lon]
    v_s = v[::step_lat, ::step_lon]
    speed_s = speed[::step_lat, ::step_lon]

    data_extent = (float(np.min(lat)), float(np.min(lon)), float(np.max(lat)), float(np.max(lon)))
    if extent is not None:
        plot_extent = parse_extent(extent) if isinstance(extent, str) else extent
    else:
        plot_extent = data_extent

    plot_cfg = plot_config or {}
    density = plot_cfg.get("density", 4)
    linewidth = plot_cfg.get("linewidth", 0.5)
    arrowsize = plot_cfg.get("arrowsize", 0.5)
    vmin = plot_cfg.get("vmin", 0.0)
    vmax = plot_cfg.get("vmax", 0.9)
    figsize = plot_cfg.get("figsize", (8, 4))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.7], projection=ccrs.Mercator())
    cplt.create(
        [plot_extent[1], plot_extent[3], plot_extent[0], plot_extent[2]],
        gridlines=True,
        ax=ax,
        oceancolor="none",
        proj=ccrs.Mercator(),
    )

    lon2d, lat2d = np.meshgrid(lon_s, lat_s)
    mag = ax.pcolormesh(
        lon2d,
        lat2d,
        speed_s,
        shading="auto",
        cmap=cmo.speed,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    ax.streamplot(
        lon2d,
        lat2d,
        u_s,
        v_s,
        color="black",
        density=density,
        linewidth=linewidth,
        arrowsize=arrowsize,
        transform=ccrs.PlateCarree(),
    )
    drift_lons = [r["lon"] for r in rows if r.get("ok", True)]
    drift_lats = [r["lat"] for r in rows if r.get("ok", True)]
    if drift_lons and drift_lats:
        ax.plot(drift_lons, drift_lats, color="black", linewidth=2.0, transform=ccrs.PlateCarree())
        ax.scatter([drift_lons[0]], [drift_lats[0]], color="lime", s=15, zorder=2, transform=ccrs.PlateCarree())
        ax.scatter([drift_lons[-1]], [drift_lats[-1]], color="red", s=15, zorder=2, transform=ccrs.PlateCarree())

    fig.colorbar(mag, ax=ax, label="current speed (m/s)")
    hours_text = f"{hours:g}h"
    if data_time:
        subtitle = f"{data_time} - {model_label} - {hours_text}"
    else:
        subtitle = f"{model_label} - {hours_text}"
    fig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.text(
        0.5,
        0.045,
        "Generated by the Glider Guidance System 3 (GGS3)",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.94, 0.9])
    fig_path = outdir / "drift_stream.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    last = rows[-1]
    print(f"rows={len(rows)}")
    print(f"endpoint: {last.get('time')} lat={last.get('lat')} lon={last.get('lon')} ok={last.get('ok')}")
    print(f"wrote: {outdir / 'drift.csv'}")
    print(f"wrote: {outdir / 'drift.geojson'}")
    print(f"wrote: {fig_path}")
    return outdir, rows, fig_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-time", required=True, help="ISO time, e.g. 2026-01-20T23:00:00Z")
    ap.add_argument("--start-lat", type=float, required=True)
    ap.add_argument("--start-lon", type=float, required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--dt", type=int, default=600)
    ap.add_argument("--depth-m", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--extent", default=None, help="lat_min,lon_min,lat_max,lon_max")
    ap.add_argument("--model", default=None, choices=["ESPC", "CMEMS", "espc", "cmems"])
    args = ap.parse_args()

    run_drift(
        start_time=args.start_time,
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        hours=args.hours,
        dt=args.dt,
        depth_m=args.depth_m,
        outdir=args.outdir,
        extent=args.extent,
        model=args.model,
    )


if __name__ == "__main__":
    main()
