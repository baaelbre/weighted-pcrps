"""Download the compact 2020 controls in resumable month/variable jobs.

Keep all requested leads in each job: FuXi stores all 60 leads in one chunk.
Six datasets x two variables x twelve months = 144 jobs, six at a time.
The final six filenames are unchanged, so the evaluators need no changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import xarray as xr

BASE = "gs://weatherbench2/datasets/"
GRID = "240x121_equiangular_with_poles_conservative.zarr"
SOURCES = {
    "era5": (BASE + "era5/1959-2023_01_10-6h-" + GRID, "era5_240x121_eval_times.zarr"),
    "hres": (BASE + "hres/2016-2022-0012-" + GRID, "hres_240x121_2020.zarr"),
    "graphcast": (BASE + "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-" + GRID, "graphcast_240x121_2020.zarr"),
    "pangu": (BASE + "pangu/2018-2022_0012_" + GRID, "pangu_240x121_2020.zarr"),
    "fuxi": (BASE + "fuxi/2020-" + GRID, "fuxi_240x121_2020.zarr"),
    "hres_fc0": (BASE + "hres_t0/2016-2022-6h-" + GRID, "ifs_analysis_240x121_eval_times.zarr"),
}
VARIABLES = ["2m_temperature", "10m_wind_speed"]


def check_coordinates(actual, expected):
    """Do not accept a different grid, time range, variable or lead list."""
    if set(actual.data_vars) != set(expected.data_vars) or actual.sizes != expected.sizes:
        raise ValueError("Variables or dimensions differ from the requested data.")
    for dim in expected.dims:
        if not np.array_equal(actual[dim].values, expected[dim].values):
            raise ValueError(f"Unexpected {dim} coordinates.")


def digest(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def completed_output(path, selected, source):
    """Reuse a final file only if its monthly checksums and metadata still match."""
    marker = path.with_suffix(".complete.json")
    if not path.exists() or not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text())
        if info["source"] != source:
            return False
        with xr.open_zarr(path, chunks=None) as ds:
            check_coordinates(ds, selected)
            for variable in VARIABLES:
                for month in range(1, 13):
                    data = ds[variable].isel(time=np.flatnonzero(ds.time.dt.month.values == month))
                    dims = [d for d in ["time", "prediction_timedelta", "latitude", "longitude"] if d in data.dims]
                    if digest(data.transpose(*dims).values) != info["sha256"][variable][str(month)]:
                        return False
        return True
    except (ValueError, KeyError, OSError):
        return False


def download_month(name, variable, month, selected, directory, source):
    """One writer per store; a checksum marks a fully written monthly part."""
    short = "t2m" if variable == "2m_temperature" else "w10"
    path = directory / name / short / f"2020-{month:02d}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix(".complete.json")
    log = path.with_suffix(".log")
    subset = selected[[variable]].isel(time=np.flatnonzero(selected.time.dt.month.values == month))
    subset = subset.transpose(*[d for d in ["time", "prediction_timedelta", "latitude", "longitude"] if d in subset.dims])
    if not subset.sizes["time"]:
        raise ValueError(f"{name}: no source timestamps in month {month}.")
    with log.open("a") as stream:
        started = time.monotonic()
        stream.write(f"\n{time.asctime()} {name} {short} month {month}; {source}\n")
        stream.flush()
        if path.exists() and marker.exists():
            try:
                info = json.loads(marker.read_text())
                with xr.open_zarr(path, chunks=None) as previous:
                    check_coordinates(previous, subset)
                    if info["source"] != source or digest(previous[variable].values) != info["sha256"]:
                        raise ValueError("Checksum or source changed.")
                stream.write("Reusing verified completed part.\n")
                return path, "cached"
            except (ValueError, KeyError, OSError):
                stream.write("Cached part does not match; downloading again.\n")
        for attempt in range(1, 4):
            try:
                # At most two Dask threads per job; no large nested worker pool.
                data = subset.compute(scheduler="threads", num_workers=2)
                data.attrs = {"source_url": source, "download_complete": "true"}
                for v in data.variables:
                    data[v].encoding = {}
                if not data[variable].attrs.get("units"):
                    data[variable].attrs["units"] = "K" if short == "t2m" else "m s-1"
                partial = path.with_name(path.name + ".partial")
                data.to_zarr(partial, mode="w", consolidated=True, zarr_format=2)
                checksum = digest(data[variable].values)
                with xr.open_zarr(partial, chunks=None) as check:
                    check_coordinates(check, data)
                    if digest(check[variable].values) != checksum:
                        raise ValueError("Written values differ from downloaded values.")
                if path.exists():
                    shutil.rmtree(path)  # Only an invalid part in our own cache.
                os.rename(partial, path)
                info = {"source": source, "sha256": checksum,
                        "nonfinite_values": int((~np.isfinite(data[variable].values)).sum())}
                marker.write_text(json.dumps(info, indent=2) + "\n")
                stream.write(f"Saved in {time.monotonic()-started:.1f}s; nonfinite values: {info['nonfinite_values']}\n")
                return path, "downloaded"
            except Exception:
                stream.write(f"Attempt {attempt}/3 failed:\n{traceback.format_exc()}\n")
                stream.flush()
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)


def finalize(name, selected, directory, output, source):
    """Merge LOCAL parts once; no concurrent appends to a common Zarr store."""
    with ExitStack() as stack:
        fields = []
        for short in ["t2m", "w10"]:
            parts = [stack.enter_context(xr.open_zarr(directory / name / short / f"2020-{month:02d}.zarr", chunks={}))
                     for month in range(1, 13)]
            fields.append(xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="equals", join="exact"))
        merged = xr.merge(fields, join="exact", compat="equals")
        check_coordinates(merged, selected)
        merged.attrs = {"source_url": source, "download_complete": "true", "year": 2020,
                        "preparation": "Month/variable jobs; 00/12 UTC; precomputed WB2 wind speed."}
        merged = merged.chunk({d: n for d, n in {"time": 31, "prediction_timedelta": 1, "latitude": -1, "longitude": -1}.items() if d in merged.dims})
        for variable in merged.variables:
            merged[variable].encoding = {}
        partial = output.with_name(output.name + ".partial")
        task = merged.to_zarr(partial, mode="w", consolidated=True, zarr_format=2, compute=False)
        task.compute(scheduler="threads", num_workers=2)
        with xr.open_zarr(partial, chunks={}) as check:
            check_coordinates(check, selected)
        backup = output.with_name(output.name + ".previous")
        if output.exists():
            if backup.exists():
                raise FileExistsError(f"Inspect previous backup before replacing {output}: {backup}")
            os.rename(output, backup)
        os.rename(partial, output)
        if backup.exists():
            shutil.rmtree(backup)
        hashes = {}
        for variable, short in zip(VARIABLES, ["t2m", "w10"]):
            hashes[variable] = {str(month): json.loads((directory/name/short/f"2020-{month:02d}.complete.json").read_text())["sha256"] for month in range(1, 13)}
        output.with_suffix(".complete.json").write_text(json.dumps({"source": source, "sha256": hashes}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["hres", "graphcast", "pangu", "fuxi"], default=["hres", "graphcast", "pangu", "fuxi"])
    parser.add_argument("--leads", nargs="+", type=int, default=[12, 48, 120, 168, 240])
    parser.add_argument("--include-hres-fc0", action="store_true")
    parser.add_argument("--jobs", type=int, default=6, help="Concurrent month/variable jobs; two Dask threads each.")
    parser.add_argument("--plan", action="store_true", help="Read source metadata and print the plan without downloading fields.")
    args = parser.parse_args()
    if args.jobs < 1 or any(lead <= 0 or lead % 6 for lead in args.leads):
        parser.error("Use positive jobs and positive leads divisible by six hours.")
    args.leads = sorted(set(args.leads))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (args.data_dir / ".prepare_control_data.lock").open("a") as lock, ExitStack() as stack:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A downloader already owns this data directory.")
        names = list(dict.fromkeys(["era5", *args.models, *(["hres_fc0"] if args.include_hres_fc0 else [])]))
        datasets, outputs, coverage = {}, {}, {}
        for name in names:
            source, filename = SOURCES[name]
            print(f"Reading metadata: {name}", flush=True)
            # Public HTTPS avoids GCS authentication/bucket-type discovery and
            # its version-dependent startup delays. TLS verification stays on.
            url = source.replace("gs://", "https://storage.googleapis.com/", 1)
            options = {"storage_options": {"client_kwargs": {"trust_env": True}}} if url.startswith("https://") else {}
            ds = stack.enter_context(xr.open_zarr(url, consolidated=True, chunks={}, decode_timedelta=True, **options))
            ds = ds[VARIABLES].rename({a: b for a, b in [("lat", "latitude"), ("lon", "longitude")] if a in ds.dims})
            ds = ds.sortby("latitude").sel(latitude=slice(-60, 90), time=slice("2020-01-01", "2020-12-31T23:59:59"))
            ds = ds.where(ds.time.dt.hour.isin([0, 12]), drop=True)
            for dim in ds.dims:
                if len(np.unique(ds[dim])) != ds.sizes[dim]:
                    raise ValueError(f"{name}: duplicate {dim} coordinates.")
            if not ds.sizes["time"] or np.isnat(ds.time.values).any():
                raise ValueError(f"{name}: invalid/empty time coordinate.")
            if "prediction_timedelta" in ds.dims:
                targets = np.array(args.leads, dtype="timedelta64[h]") if np.issubdtype(ds.prediction_timedelta.dtype, np.timedelta64) else args.leads
                ds = ds.sel(prediction_timedelta=targets)
            datasets[name], outputs[name] = ds, args.data_dir / filename
            expected = np.arange(np.datetime64("2020-01-01T00", "h"), np.datetime64("2021-01-01T00", "h"), np.timedelta64(12, "h"))
            missing = np.setdiff1d(expected, ds.time.values.astype("datetime64[h]"))
            coverage[name] = {"source": source, "n_times": ds.sizes["time"], "missing_00_12_timestamps": missing.astype(str).tolist(),
                              "latitude_count": ds.sizes["latitude"], "southern_latitude": float(ds.latitude.min()),
                              "requested_leads_hours": args.leads if "prediction_timedelta" in ds.dims else []}
            print(f"  {ds.sizes['time']}/732 source timestamps; compact output {ds.nbytes/1e9:.3f} GB", flush=True)
        print(f"Plan: {len(names)*24} month/variable jobs, {args.jobs} concurrent; leads {args.leads} h.", flush=True)
        print("Progress is completed jobs, not bytes or an exact remaining-time estimate.", flush=True)
        print("Missing source timestamps are reported, never fabricated. Evaluation uses common valid dates.", flush=True)
        if args.plan:
            return
        directory = args.data_dir / "download_parts" / ("leads_" + "-".join(map(str, args.leads)))
        directory.mkdir(parents=True, exist_ok=True)
        failures = []
        pending = []
        previous_coverage = args.data_dir / "download_coverage.json"
        previous_coverage = json.loads(previous_coverage.read_text()) if previous_coverage.exists() else {}
        for name in names:
            if completed_output(outputs[name], datasets[name], SOURCES[name][0]):
                print(f"Keeping verified final store: {outputs[name]}", flush=True)
                coverage[name]["nonfinite_values"] = previous_coverage.get(name, {}).get("nonfinite_values", "See final store; previous coverage report unavailable.")
            else:
                pending.append(name)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            tasks = {pool.submit(download_month, name, var, month, datasets[name], directory, SOURCES[name][0]): (name, var, month)
                     for month in range(1, 13) for name in pending for var in VARIABLES}
            for count, future in enumerate(as_completed(tasks), start=1):
                key = tasks[future]
                try:
                    _, status = future.result()
                except Exception as error:
                    status = f"FAILED: {error}"
                    failures.append(key)
                print(f"[{count:3d}/{len(tasks)}; {100*count/len(tasks):5.1f}%] {key[0]} {key[1]} month {key[2]:02d}: {status}; elapsed {(time.monotonic()-started)/60:.1f} min", flush=True)
        if failures:
            raise SystemExit(f"{len(failures)} jobs failed. Details: {directory}. Rerun the same command to reuse completed jobs.")
        for name in pending:
            print(f"Merging local parts: {name}", flush=True)
            output = outputs[name]
            if output.exists():
                with xr.open_zarr(output, chunks={}) as existing:
                    if existing.attrs.get("source_url") not in [None, SOURCES[name][0]]:
                        raise ValueError(f"Refusing to replace {output} from a different source.")
            finalize(name, datasets[name], directory, output, SOURCES[name][0])
            coverage[name]["nonfinite_values"] = {}
            for short in ["t2m", "w10"]:
                coverage[name]["nonfinite_values"][short] = sum(json.loads((directory/name/short/f"2020-{month:02d}.complete.json").read_text())["nonfinite_values"] for month in range(1, 13))
            print(f"Saved {output}; nonfinite counts {coverage[name]['nonfinite_values']}", flush=True)
        report = args.data_dir / "download_coverage.json"
        report.write_text(json.dumps(coverage, indent=2) + "\n")
        print(f"Done in {(time.monotonic()-started)/60:.1f} min. Coverage: {report}", flush=True)
        print("Monthly parts are retained for resume. Remove download_parts only after checking the final stores.", flush=True)


if __name__ == "__main__":
    main()
