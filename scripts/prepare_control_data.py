"""Download compact 2020 inputs for the controls, not full WeatherBench archives.

Only T2m, precomputed W10, 00/12 UTC and requested leads on the supplied 1.5
degree grid. Source paths: WeatherBench 2 Data Guide. No new model inference.
"""
import argparse
import os
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--models", nargs="+", choices=["hres", "graphcast", "pangu", "fuxi"], default=["hres", "graphcast", "pangu", "fuxi"])
parser.add_argument("--leads", nargs="+", type=int, default=[48, 120, 240])
parser.add_argument("--include-hres-fc0", action="store_true")
args = parser.parse_args()
if any(lead <= 0 or lead % 6 for lead in args.leads):
    parser.error("Leads must be positive multiples of six hours.")
args.data_dir.mkdir(parents=True, exist_ok=True)
base = "gs://weatherbench2/datasets/"
grid = "240x121_equiangular_with_poles_conservative.zarr"
sources = {
    "era5": (base + "era5/1959-2023_01_10-6h-" + grid, "era5_240x121_eval_times.zarr"),
    "hres": (base + "hres/2016-2022-0012-" + grid, "hres_240x121_2020.zarr"),
    "graphcast": (base + "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-" + grid, "graphcast_240x121_2020.zarr"),
    "pangu": (base + "pangu/2018-2022_0012_" + grid, "pangu_240x121_2020.zarr"),
    "fuxi": (base + "fuxi/2020-" + grid, "fuxi_240x121_2020.zarr"),
    "hres_fc0": (base + "hres_t0/2016-2022-6h-" + grid, "ifs_analysis_240x121_eval_times.zarr"),
}
variables = ["2m_temperature", "10m_wind_speed"]
for name in ["era5", *args.models, *(["hres_fc0"] if args.include_hres_fc0 else [])]:
    source, filename = sources[name]
    path = args.data_dir / filename
    if path.exists():
        with xr.open_zarr(path) as ds:
            if not all(v in ds for v in variables):
                raise ValueError(f"Existing {path} lacks required variables; use a new data directory.")
            if name not in ["era5", "hres_fc0"]:
                lead_values = ds.prediction_timedelta.values
                hours = lead_values / np.timedelta64(1, "h") if np.issubdtype(lead_values.dtype, np.timedelta64) else lead_values
                if not set(args.leads).issubset(hours):
                    raise ValueError(f"Existing {path} lacks requested leads; use a new data directory for the expanded download.")
            if not np.any(ds.time.dt.year == 2020):
                raise ValueError(f"Existing {path} has no 2020 data.")
        print(f"Keeping {path}", flush=True)
        continue
    print(f"Downloading {name}: {source}", flush=True)
    with xr.open_zarr(source, storage_options={"token": "anon"}, chunks={}, decode_timedelta=True) as ds:
        ds = ds[variables].rename({a: b for a, b in [("lat", "latitude"), ("lon", "longitude")] if a in ds.dims})
        ds = ds.sortby("latitude").sel(latitude=slice(-60, 90), time=slice("2020-01-01", "2020-12-31T23:59:59"))
        ds = ds.where(ds.time.dt.hour.isin([0, 12]), drop=True)
        if "prediction_timedelta" in ds.dims:
            targets = np.array(args.leads, dtype="timedelta64[h]") if np.issubdtype(ds.prediction_timedelta.dtype, np.timedelta64) else args.leads
            ds = ds.sel(prediction_timedelta=targets)
        ds.attrs.update(source_url=source, requested_leads_hours=",".join(map(str, args.leads)), preparation="Compact ProbEx controls; precomputed WeatherBench wind speed.")
        for v in variables:
            if not ds[v].attrs.get("units"):
                ds[v].attrs["units"] = "K" if v == "2m_temperature" else "m s-1"
        ds = ds.chunk({d: c for d, c in {"time": 31, "latitude": 30, "longitude": 60, "prediction_timedelta": 1}.items() if d in ds.dims})
        for v in ds.variables:
            ds[v].encoding = {}
        partial = path.with_name(path.name + ".partial")
        print("Progress counts Dask tasks, not downloaded bytes.", flush=True)
        with dask.config.set(scheduler="threads", num_workers=4), ProgressBar(dt=1, minimum=3):
            ds.to_zarr(partial, mode="w", consolidated=True, zarr_format=2)
        with xr.open_zarr(partial) as check:
            if check.sizes["time"] != ds.sizes["time"] or not all(v in check for v in variables):
                raise ValueError(f"Output validation failed: {partial}")
        os.rename(partial, path)
    print(f"Saved {path}", flush=True)
