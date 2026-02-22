from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Tuple
import math
from datetime import datetime, timedelta

EARTH_R = 6371000.0  # meters


@dataclass
class DriftConfig:
    start_time: datetime          # timezone-aware (UTC preferred)
    start_lat: float
    start_lon: float
    duration_s: int               # total sim duration (seconds)
    dt_s: int = 600               # timestep seconds
    depth_m: Optional[float] = None
    stop_on_nan: bool = True


def _advect_spherical(lat_deg: float, lon_deg: float, u: float, v: float, dt_s: int) -> Tuple[float, float]:
    """
    Small-step spherical update.
    u east (m/s), v north (m/s). Returns (lat, lon) degrees.
    """
    lat_rad = math.radians(lat_deg)
    dy = v * dt_s
    dx = u * dt_s
    dlat = (dy / EARTH_R) * (180.0 / math.pi)
    coslat = max(1e-6, math.cos(lat_rad))
    dlon = (dx / (EARTH_R * coslat)) * (180.0 / math.pi)
    return lat_deg + dlat, lon_deg + dlon


def simulate_drift(
    cfg: DriftConfig,
    sample_uv: Callable[[datetime, float, float, Optional[float]], Tuple[float, float]],
) -> List[Dict]:
    if cfg.start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware (UTC).")

    steps = max(1, int(cfg.duration_s // cfg.dt_s))
    t = cfg.start_time
    lat = float(cfg.start_lat)
    lon = float(cfg.start_lon)

    out: List[Dict] = []
    for k in range(steps + 1):
        u, v = sample_uv(t, lat, lon, cfg.depth_m)

        bad = (u is None or v is None or
               (isinstance(u, float) and math.isnan(u)) or
               (isinstance(v, float) and math.isnan(v)))

        if bad:
            out.append({"time": t.isoformat(), "lat": lat, "lon": lon, "u": u, "v": v, "ok": False})
            if cfg.stop_on_nan:
                break
            t = t + timedelta(seconds=cfg.dt_s)
            continue

        speed = math.hypot(u, v)
        heading_deg = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0  # 0=N, 90=E

        out.append({
            "time": t.isoformat(),
            "lat": lat,
            "lon": lon,
            "u": float(u),
            "v": float(v),
            "speed_mps": float(speed),
            "heading_deg": float(heading_deg),
            "ok": True,
        })

        if k == steps:
            break

        lat, lon = _advect_spherical(lat, lon, u, v, cfg.dt_s)
        t = t + timedelta(seconds=cfg.dt_s)

    return out
