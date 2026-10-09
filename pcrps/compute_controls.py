"""Paired EasyUQ controls using the shared ProbEx data directory.

Default: reproduce the in-sample potential-information setting. --mode holdout
fits only dates up to --calibration-end and requires later evaluation dates.
Neither mode is called native probabilistic-model skill or causal extrapolation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import dask
import numpy as np
import xarray as xr
from pcrps.control_scores import REPRESENTATIONS, SCORES, score_cell

VARIABLES = {"t2m": "2m_temperature", "w10": "10m_wind_speed"}
MODEL_DIRS = {"hres": "HRES", "graphcast": "GraphCast", "pangu": "Pangu", "fuxi": "FuXi"}


def check_units(data, variable, required=False):
    units = str(data.attrs.get("units") or "").lower().replace(" ", "").replace("**", "^")
    allowed = {"k", "kelvin"} if variable == "t2m" else {"m/s", "ms-1", "ms^-1"}
    if (required and not units) or (units and units not in allowed):
        raise ValueError(f"Unexpected {variable} units: {data.attrs.get('units')!r}.")


def normalize(data):
    data = data.rename({a: b for a, b in [("lat", "latitude"), ("lon", "longitude")] if a in data.dims})
    data = data.assign_coords(longitude=(data.longitude + 180) % 360 - 180)
    for dim in ["latitude", "longitude"]:
        data = data.assign_coords({dim: np.round(data[dim].values.astype(float), 6)}).sortby(dim)
        if not np.isfinite(data[dim]).all() or len(np.unique(data[dim])) != data.sizes[dim]:
            raise ValueError(f"Invalid {dim} coordinates.")
    # Same boundary for forecasts, truth, records/scales and land-sea mask.
    # Rounding above removes coordinate noise; this does not interpolate.
    return data.sel(latitude=data.latitude[(data.latitude > -60) & (data.latitude <= 90)])


def field_at_lead(ds, var, lead):
    field = normalize(ds[var])
    td = field.prediction_timedelta
    target = np.timedelta64(lead, "h") if np.issubdtype(td.dtype, np.timedelta64) else lead
    field = field.sel(prediction_timedelta=target, drop=True)
    field = field.assign_coords(time=field.time.values.astype("datetime64[ns]") + np.timedelta64(lead, "h"))
    if set(field.dims) != {"time", "latitude", "longitude"}:
        raise ValueError(f"Expected a deterministic field, found {field.dims}.")
    return field.sortby("time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("PCRPS_DATA_DIR", "data"),
                        help="Shared ProbEx root containing era5/, forecasts/ and era5_records/.")
    parser.add_argument("--records-root", type=Path, required=True, help="ProbEx record + scale NetCDF files for the historical baseline.")
    parser.add_argument("--output", type=Path, default=Path("results/controls"))
    parser.add_argument("--models", nargs="+", choices=["hres", "graphcast", "pangu", "fuxi"], default=["hres", "graphcast", "pangu", "fuxi"])
    parser.add_argument("--var", choices=list(VARIABLES), default="t2m")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--record-tag", default="1979_2019")
    parser.add_argument("--leads", nargs="+", type=int, default=[48, 120, 240])
    parser.add_argument("--depths", nargs="+", type=float, default=[0, 0.5, 1, 2])
    parser.add_argument("--surface", choices=["land", "sea", "all"], default="land")
    parser.add_argument("--truth", choices=["ERA5", "HRES_fc0"], default="ERA5")
    parser.add_argument("--mode", choices=["in-sample", "holdout"], default="in-sample")
    parser.add_argument("--calibration-end", help="Last valid date available to IDR; required for holdout.")
    parser.add_argument("--eval-start", help="Optional common evaluation start; use to compare modes on identical dates.")
    parser.add_argument("--eval-end", help="Optional common evaluation end (inclusive date).")
    parser.add_argument("--pilot-cells", type=int, help="Use this many geographically spread eligible cells; keep full time series for fitting.")
    args = parser.parse_args()
    first_year, last_year = map(int, args.record_tag.split("_"))
    if not first_year < last_year < args.year or min(args.leads) <= 0 or min(args.depths) < 0 or not np.isfinite(args.depths).all():
        parser.error("Require a historical baseline, positive leads and finite nonnegative depths.")
    if args.mode == "holdout" and not args.calibration_end:
        parser.error("Holdout needs --calibration-end; no automatic split is assumed.")
    if args.pilot_cells is not None and args.pilot_cells < 1:
        parser.error("--pilot-cells must be positive.")
    args.models = list(dict.fromkeys(args.models))
    args.leads, args.depths = sorted(set(args.leads)), sorted(set(args.depths))
    args.output.mkdir(parents=True, exist_ok=True)
    variable = VARIABLES[args.var]
    truth_folder = "era5" if args.truth == "ERA5" else "forecasts/HRES_fc0"
    truth_file = args.data_dir / truth_folder / f"{args.truth}_{args.year}_0012_no_Antarctic_1p50.zarr"
    forecast_files = {model: args.data_dir / "forecasts" / MODEL_DIRS[model] / f"{model}_240x121_{args.year}.zarr"
                      for model in args.models}
    for path in [truth_file, *forecast_files.values()]:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {path}. --data-dir must be the shared ProbEx data root.")
    print(f"Data: {args.data_dir}; truth: {args.truth}; records: {args.record_tag}", flush=True)
    print(f"Models: {args.models}; leads: {args.leads}; mode: {args.mode}", flush=True)
    truth_ds = xr.open_zarr(truth_file, chunks={})
    truth = normalize(truth_ds[variable]).sel(time=slice(f"{args.year}-01-01", f"{args.year}-12-31T23:59:59"))
    check_units(truth, args.var)
    truth = truth.where(truth.time.dt.hour.isin([0, 12]), drop=True)
    sources = {model: xr.open_zarr(path, chunks={}) for model, path in forecast_files.items()}
    times = truth.time.values
    for ds in [truth, *sources.values()]:
        if len(np.unique(ds.time)) != ds.sizes["time"] or np.isnat(ds.time).any():
            raise ValueError("Duplicate or missing timestamps.")
    # The intersection covers all models AND all requested leads.
    for model, ds in sources.items():
        for lead in args.leads:
            field = field_at_lead(ds, variable, lead)
            check_units(field, args.var)
            for dim in ["latitude", "longitude"]:
                if not np.array_equal(field[dim], truth[dim]):
                    raise ValueError(f"{model}: {dim} differs; regrid before scoring.")
            times = np.intersect1d(times, field.time.values)
    if not len(times):
        raise ValueError("No common valid times.")
    truth = truth.sel(time=times)
    print(f"Common grid: {truth.sizes['latitude']} x {truth.sizes['longitude']}; -60 < latitude <= 90.", flush=True)
    print(f"Common valid times across all models/leads: {len(times)} ({times[0]} to {times[-1]}).", flush=True)
    evaluate = np.ones(len(times), dtype=bool)
    if args.eval_start:
        evaluate &= times >= np.datetime64(args.eval_start)
    if args.eval_end:
        evaluate &= times < np.datetime64(args.eval_end, "D") + np.timedelta64(1, "D")
    if args.mode == "holdout":
        boundary = np.datetime64(args.calibration_end, "D") + np.timedelta64(1, "D")
        fit = times < boundary
        if args.eval_start and np.datetime64(args.eval_start) < boundary:
            parser.error("Holdout evaluation must start after calibration ends.")
        evaluate &= times >= boundary
    else:
        fit = evaluate.copy()
    if fit.sum() < 20 or not evaluate.any():
        raise ValueError("Need at least 20 calibration cases and nonempty verification cases.")
    months = truth.time.dt.month.values - 1
    cases = ["max", "min"] if args.var == "t2m" else ["max"]
    names = ["heat", "cold"] if args.var == "t2m" else ["wind"]
    signs = [1, -1] if args.var == "t2m" else [1]
    histories = []
    with xr.open_dataset(args.records_root / "ERA5_lsm_sfc.nc") as source:
        lsm = normalize(source.lsm.squeeze(drop=True)).transpose("latitude", "longitude").load()
    xr.align(lsm, truth, join="exact")
    if not np.isfinite(lsm).all():
        raise ValueError("Nonfinite land-sea mask.")
    eligible = np.ones(lsm.shape, dtype=bool) if args.surface == "all" else (lsm.values > 0.5 if args.surface == "land" else lsm.values <= 0.5)
    for case in cases:
        path = args.records_root / f"ERA5_record_{case}_month_{args.record_tag}_{args.var}.nc"
        with xr.open_dataset(path) as source:
            history = normalize(source).load()
        if "scale" not in history:
            with xr.open_dataset(args.records_root / f"ERA5_record_scale_{case}_month_{args.record_tag}_{args.var}.nc") as source:
                combined = normalize(source).load()
            original = history["record"] if "record" in history else history[list(history.data_vars)[0]]
            original, updated = xr.align(original, combined.record, join="exact")
            if not np.array_equal(original.transpose(*updated.dims), updated):
                raise ValueError("Record and record/scale files disagree.")
            history = combined
        history = history.transpose("month", "latitude", "longitude")
        xr.align(history, truth, join="exact")
        for key, expected in {"record_tag": args.record_tag, "record_start_year": first_year, "record_end_year": last_year,
                              "record_hours_utc": "0,12", "scale_ddof": 1, "scale_detrended": "false",
                              "n_years": last_year-first_year+1, "variable": args.var, "extreme": case}.items():
            if str(history.attrs.get(key)) != str(expected):
                raise ValueError(f"{path}: missing/inconsistent {key}.")
        if not np.array_equal(history.month, np.arange(1, 13)):
            raise ValueError("Expected all 12 months.")
        for name in ["record", "scale"]:
            check_units(history[name], args.var, required=True)
        if not np.isfinite(history.record).all() or not np.isfinite(history.scale).all() or (history.scale < 0).any():
            raise ValueError("Invalid historical records/scales.")
        eligible &= (history.scale > 0).all("month").values
        histories.append(history)
    cells = np.flatnonzero(eligible.ravel())
    eligible_cell_count = len(cells)
    if args.pilot_cells and len(cells) > args.pilot_cells:
        cells = cells[np.linspace(0, len(cells)-1, args.pilot_cells, dtype=int)]
    if not len(cells):
        raise ValueError("No eligible cells.")
    lat_index, lon_index = np.unravel_index(cells, lsm.shape)
    # Vectorized indexing reads only pilot cells when requested, with all times.
    select = {"latitude": xr.DataArray(lat_index, dims="cell"), "longitude": xr.DataArray(lon_index, dims="cell")}
    reference = hashlib.sha256()
    for array in [times, lsm.latitude.values, lsm.longitude.values, cells, *[h.record.values for h in histories], *[h.scale.values for h in histories]]:
        reference.update(np.ascontiguousarray(array).tobytes())
    records = np.stack([h.record.isel(select).transpose("month", "cell").values for h in histories])
    scales = np.stack([h.scale.isel(select).transpose("month", "cell").values for h in histories])
    weight = np.maximum(np.cos(np.deg2rad(lsm.latitude.values[lat_index])), 0)
    with dask.config.set(scheduler="threads", num_workers=4):
        observations = truth.isel(select).transpose("time", "cell").compute().values.astype(float)
    if not np.isfinite(observations).all():
        raise ValueError("Nonfinite truth; do not silently change matched cases.")
    print(f"{args.mode}: {fit.sum()} fitting times, {evaluate.sum()} verification times; {len(cells)}/{eligible_cell_count} eligible cells.", flush=True)
    print("All four representations use the same cases; four CPU processes, no GPU.", flush=True)
    for lead in args.leads:
        start = time.monotonic()
        fields = [field_at_lead(sources[m], variable, lead).sel(time=times).isel(select).transpose("time", "cell") for m in args.models]
        with dask.config.set(scheduler="threads", num_workers=4):
            forecasts = np.stack([v.values for v in dask.compute(*fields)]).astype(float)
        if not np.isfinite(forecasts).all():
            raise ValueError("Nonfinite forecast; all models must use identical cases.")
        totals = np.zeros((len(args.models), len(cases), 4, len(SCORES), len(args.depths)))
        event_totals = np.zeros_like(totals)
        counts = np.zeros((len(cases), len(args.depths)), dtype=np.int64)
        event_weight = np.zeros_like(counts, dtype=float)
        calibration_max = np.zeros((len(args.models), len(cells)))
        jobs = ((forecasts[:, :, cell], observations[:, cell], months, records[:, :, cell], scales[:, :, cell],
                 args.depths, signs, fit, evaluate) for cell in range(len(cells)))
        with ProcessPoolExecutor(max_workers=4) as pool:
            for cell, (total, event_total, count, fit_max) in enumerate(pool.map(score_cell, jobs, chunksize=4)):
                totals += weight[cell] * total
                event_totals += weight[cell] * event_total
                counts += count
                event_weight += weight[cell] * count
                calibration_max[:, cell] = fit_max
                if (cell+1) % 25 == 0 or cell+1 == len(cells):
                    elapsed = time.monotonic()-start
                    remaining = elapsed / (cell+1) * (len(cells)-cell-1)
                    print(f"{lead} h: {cell+1}/{len(cells)} cells; elapsed {elapsed/60:.1f} min, rough remaining {remaining/60:.1f} min", flush=True)
        all_weight = weight.sum() * evaluate.sum()
        dims = ["model", "event", "representation", "score", "depth"]
        coords = dict(model=args.models, event=names, representation=REPRESENTATIONS, score=SCORES, depth=args.depths)
        ds = xr.Dataset(coords=coords)
        ds["all"] = (dims, totals / all_weight)
        ds["record_contribution"] = (dims, event_totals / all_weight)
        ds["nonrecord_contribution"] = (dims, (totals-event_totals) / all_weight)
        ds["record_weight"] = (["event", "depth"], event_weight)
        ds["record_count"] = (["event", "depth"], counts)
        ds["record"] = ds.record_contribution * all_weight / ds.record_weight.where(ds.record_weight > 0)
        ds["nonrecord"] = ds.nonrecord_contribution * all_weight / (all_weight-ds.record_weight).where(all_weight > ds.record_weight)
        ds["observed_probability"] = ds.record_weight / all_weight
        ds.attrs = dict(mode=args.mode, year=args.year, variable=args.var, record_tag=args.record_tag,
                        truth=args.truth, truth_file=str(truth_file.resolve()), surface=args.surface,
                        data_directory=str(args.data_dir.resolve()), latitude_domain="-60 < latitude <= 90",
                        forecast_files="; ".join(f"{m}: {p.resolve()}" for m, p in forecast_files.items()),
                        reference_fingerprint=reference.hexdigest(), valid_count=int(evaluate.sum()*len(cells)),
                        valid_weight=float(all_weight), calibration_time_count=int(fit.sum()), eligible_cell_count=eligible_cell_count,
                        selected_cell_count=len(cells), pilot="true" if args.pilot_cells else "false",
                        scoring="Exact weighted distributions, no fair correction; record/nonrecord contributions sum to all.",
                        interpretation="In-sample potential-information diagnostic" if args.mode == "in-sample" else "Chronological holdout within the available forecast year")
        stem = f"controls_{args.var}_{args.year}_{args.mode}_{args.surface}_{lead:03d}h"
        ds.to_dataframe().reset_index().to_csv(args.output / f"{stem}.csv", index=False)
        ds = ds.assign_coords(verification_time=times[evaluate], calibration_time=times[fit], cell=cells)
        ds["calibration_max"] = (["model", "cell"], calibration_max)
        ds["cell_latitude"] = ("cell", lsm.latitude.values[lat_index])
        ds["cell_longitude"] = ("cell", lsm.longitude.values[lon_index])
        ds = ds.expand_dims(leadtime=[lead])
        path = args.output / f"{stem}.nc"
        ds.to_netcdf(path.with_suffix(".nc.partial"), engine="netcdf4")
        os.replace(path.with_suffix(".nc.partial"), path)
        print(f"Saved {path}", flush=True)
    truth_ds.close()
    for source in sources.values():
        source.close()


if __name__ == "__main__":
    main()
