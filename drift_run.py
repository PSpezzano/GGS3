#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

from ggs3.cli_drift import parse_time, run_drift


def _parse_start_time(value: str) -> datetime:
    if value is None:
        raise ValueError("START_TIME is required.")
    key = value.strip().lower()
    if key in {"today", "tomorrow", "yesterday"}:
        base = date.today()
        if key == "tomorrow":
            base = base + timedelta(days=1)
        elif key == "yesterday":
            base = base - timedelta(days=1)
        return datetime(base.year, base.month, base.day, tzinfo=timezone.utc)
    return parse_time(value)


def _get_extent(config: dict):
    subset = config.get("SUBSET", {})
    extent = subset.get("EXTENT", {})
    sw = extent.get("SW_POINT")
    ne = extent.get("NE_POINT")
    if sw and ne:
        return (float(sw[0]), float(sw[1]), float(ne[0]), float(ne[1]))
    return None


def _get_start_point(config: dict):
    if "START_LAT" in config and "START_LON" in config:
        return float(config["START_LAT"]), float(config["START_LON"])
    pathfinding = config.get("PATHFINDING", {})
    waypoints = pathfinding.get("WAYPOINTS", [])
    if waypoints:
        return float(waypoints[0][0]), float(waypoints[0][1])
    raise ValueError("START_LAT/START_LON or PATHFINDING.WAYPOINTS[0] required.")


def _get_plot_config(config: dict):
    plot_cfg = config.get("PLOT", {}) or {}
    return {
        "density": plot_cfg.get("DENSITY", plot_cfg.get("density", 4)),
        "linewidth": plot_cfg.get("LINEWIDTH", plot_cfg.get("linewidth", 0.5)),
        "arrowsize": plot_cfg.get("ARROWSIZE", plot_cfg.get("arrowsize", 0.5)),
        "vmin": plot_cfg.get("VMIN", plot_cfg.get("vmin", 0.0)),
        "vmax": plot_cfg.get("VMAX", plot_cfg.get("vmax", 0.9)),
        "figsize": tuple(plot_cfg.get("FIGSIZE", plot_cfg.get("figsize", (8, 4)))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to drift run JSON config.")
    args = ap.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    start_time = config.get("START_TIME") or config.get("SUBSET", {}).get("TIME", {}).get("START_DATE")
    t0 = _parse_start_time(start_time)
    start_lat, start_lon = _get_start_point(config)
    hours = float(config.get("HOURS", config.get("DURATION_HOURS")))
    dt = int(config.get("DT", config.get("DT_SECONDS", 600)))
    depth_m = config.get("DEPTH_M", None)
    outdir = config.get("OUTDIR", None)
    dataset_path = config.get("DATASET_PATH", None)
    model = config.get("MODEL", None)
    extent = _get_extent(config)
    plot_config = _get_plot_config(config)

    run_drift(
        start_time=t0,
        start_lat=start_lat,
        start_lon=start_lon,
        hours=hours,
        dt=dt,
        depth_m=depth_m,
        outdir=outdir,
        extent=extent,
        dataset_path=dataset_path,
        model=model,
        plot_config=plot_config,
    )


if __name__ == "__main__":
    main()
