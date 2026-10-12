"""Raw forecasts: independently switch record selection and error reference.

E selects the cases; T supplies the observations used in forecast - observation.
All four E/T combinations use the same models, dates, grid and historical R.
No EasyUQ fitting, inference, resampling or forecast downloads take place here.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

import dask
from dask.diagnostics import ProgressBar
import numpy as np
import xarray as xr

from pcrps.compute_controls import VARIABLES, MODEL_DIRS, check_units, normalize, field_at_lead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get(
        "PCRPS_DATA_DIR", "/data/nvme1/bastiaan/probex-data"))
    parser.add_argument("--records-root", type=Path, default=os.environ.get("PCRPS_RECORDS_ROOT"))
    parser.add_argument("--output", type=Path, default=Path("results/reference_comparison"))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_DIRS), default=list(MODEL_DIRS))
    parser.add_argument("--var", choices=list(VARIABLES), default="t2m")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--record-tag", default="1979_2019")
    parser.add_argument("--record-hours", nargs="+", type=int, default=[0, 12],
                        help="Expected sampling of the historical record file, not the verification hours.")
    parser.add_argument("--leads", nargs="+", type=int, default=[48, 120, 240])
    parser.add_argument("--axes", nargs="+", choices=["margin", "depth"], default=["margin", "depth"])
    parser.add_argument("--margins", nargs="+", type=float, default=[0, 0.5, 1, 2, 3, 4, 5, 6],
                        help="Cumulative physical exceedance thresholds, in K or m/s.")
    parser.add_argument("--depths", nargs="+", type=float, default=[0, 0.5, 1, 2])
    parser.add_argument("--surface", choices=["land", "sea", "all"], default="land")
    parser.add_argument("--pilot-cells", type=int, help="Execution check on this many spread-out cells, retaining all dates.")
    args = parser.parse_args()
    args.records_root = args.records_root or args.data_dir / "era5_records"
    args.models = list(dict.fromkeys(args.models))
    args.axes = list(dict.fromkeys(args.axes))
    args.leads = sorted(set(args.leads))
    args.margins, args.depths = sorted(set(args.margins)), sorted(set(args.depths))
    args.record_hours = sorted(set(args.record_hours))
    try:
        first_year, last_year = map(int, args.record_tag.split("_"))
    except ValueError:
        parser.error("--record-tag must be START_END, e.g. 1979_2019.")
    if not first_year < last_year < args.year:
        parser.error("Records must end before the verification year.")
    if any(lead <= 0 or lead % 12 for lead in args.leads):
        parser.error("Use positive leads in multiples of 12 h for 00/12 UTC verification.")
    if any(not np.isfinite(k) or k < 0 for k in args.margins + args.depths):
        parser.error("Margins/depths must be finite and nonnegative.")
    if any(h not in [0, 6, 12, 18] for h in args.record_hours):
        parser.error("Historical record hours must be drawn from 0, 6, 12, 18.")
    if args.pilot_cells is not None and args.pilot_cells < 1:
        parser.error("--pilot-cells must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)

    variable = VARIABLES[args.var]
    references = ["ERA5", "HRES_fc0"]
    reference_paths = {
        "ERA5": args.data_dir / "era5" / f"ERA5_{args.year}_0012_no_Antarctic_1p50.zarr",
        "HRES_fc0": args.data_dir / "forecasts/HRES_fc0" / f"HRES_fc0_{args.year}_0012_no_Antarctic_1p50.zarr",
    }
    forecast_paths = {m: args.data_dir / "forecasts" / MODEL_DIRS[m] / f"{m}_240x121_{args.year}.zarr"
                      for m in args.models}
    print(f"Raw forecast reference comparison: {args.var}, {args.year}, {args.surface}", flush=True)
    print(f"Data: {args.data_dir}; records: {args.record_tag}, hours {args.record_hours}", flush=True)
    print(f"Models: {args.models}; leads: {args.leads}; axes: {args.axes}", flush=True)
    print("Four combinations: E=ERA5/HRES_fc0 selects events; T=ERA5/HRES_fc0 verifies them.", flush=True)

    with ExitStack() as stack:
        # 1. Open the six LOCAL stores and put all fields on the same coordinates.
        sources = {}
        for name, path in {**reference_paths, **forecast_paths}.items():
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {path}. --data-dir is the shared ProbEx data root.")
            ds = stack.enter_context(xr.open_zarr(path, chunks={}))
            if not np.issubdtype(ds.time.dtype, np.datetime64):
                raise ValueError(f"{name}: timestamps must be decoded datetimes.")
            if np.isnat(ds.time).any() or len(np.unique(ds.time)) != ds.sizes["time"]:
                raise ValueError(f"{name}: duplicate or missing timestamps.")
            sources[name] = ds
        truth = {}
        for name in references:
            da = normalize(sources[name][variable]).sortby("time")
            da = da.sel(time=slice(f"{args.year}-01-01", f"{args.year}-12-31T23:59:59"))
            da = da.sel(time=da.time.dt.hour.isin([0, 12]))
            if set(da.dims) != {"time", "latitude", "longitude"}:
                raise ValueError(f"{name}: expected time, latitude, longitude; got {da.dims}.")
            check_units(da, args.var, required=True)
            truth[name] = da
            print(f"  {name}: {da.sizes['time']} reference times in the requested year.", flush=True)
        grid = truth["ERA5"].isel(time=0, drop=True)
        # Exact alignment of spatial coordinates: no nearest neighbour matching.
        xr.align(grid, truth["HRES_fc0"], join="exact")
        times = np.intersect1d(truth["ERA5"].time, truth["HRES_fc0"].time)
        for model in args.models:
            for lead in args.leads:
                da = field_at_lead(sources[model], variable, lead)
                check_units(da, args.var, required=True)
                xr.align(grid, da, join="exact")
                times = np.intersect1d(times, da.time.values)
        if not len(times):
            raise ValueError("No common valid times across both references, all models and all requested leads.")
        print(f"Common dates: {len(times)}, from {times[0]} to {times[-1]}.", flush=True)
        print(f"Common grid: {grid.sizes['latitude']} x {grid.sizes['longitude']}; -60 < latitude <= 90.", flush=True)

        # 2. Read the fixed ERA5 monthly records (and scales for the depth axis).
        with xr.open_dataset(args.records_root / "ERA5_lsm_sfc.nc") as ds:
            lsm = normalize(ds.lsm.squeeze(drop=True)).transpose("latitude", "longitude").load()
        xr.align(grid, lsm, join="exact")
        if not np.isfinite(lsm).all() or ((lsm < 0) | (lsm > 1)).any():
            raise ValueError("Invalid land-sea mask.")
        eligible = np.ones(lsm.shape, dtype=bool)
        if args.surface == "land":
            eligible &= lsm.values > 0.5
        elif args.surface == "sea":
            eligible &= lsm.values <= 0.5
        events = {"heat": ("max", 1), "cold": ("min", -1)} if args.var == "t2m" else {"wind": ("max", 1)}
        histories, history_paths = {}, {}
        for event, (extreme, sign) in events.items():
            path = args.records_root / f"ERA5_record_{extreme}_month_{args.record_tag}_{args.var}.nc"
            with xr.open_dataset(path) as ds:
                history = normalize(ds).load()
            if "record" not in history:
                raise ValueError(f"{path}: expected a variable named record.")
            if "depth" in args.axes and "scale" not in history:
                scale_path = args.records_root / f"ERA5_record_scale_{extreme}_month_{args.record_tag}_{args.var}.nc"
                with xr.open_dataset(scale_path) as ds:
                    combined = normalize(ds).load()
                a, b = xr.align(history.record, combined.record, join="exact")
                if not np.array_equal(a.transpose(*b.dims), b):
                    raise ValueError("Separate record and record/scale files disagree.")
                history, path = combined, scale_path
            history = history.transpose("month", "latitude", "longitude")
            xr.align(grid, history, join="exact")
            expected = {"record_tag": args.record_tag, "record_start_year": first_year,
                        "record_end_year": last_year, "record_hours_utc": ",".join(map(str, args.record_hours)),
                        "variable": args.var, "extreme": extreme}
            if "depth" in args.axes:
                expected.update(scale_ddof=1, scale_detrended="false", n_years=last_year-first_year+1)
            for key, value in expected.items():
                if str(history.attrs.get(key)) != str(value):
                    raise ValueError(f"{path}: expected {key}={value!r}, found {history.attrs.get(key)!r}.")
            if not np.array_equal(history.month, np.arange(1, 13)):
                raise ValueError(f"{path}: expected months 1 through 12.")
            check_units(history.record, args.var, required=True)
            if not np.isfinite(history.record).all():
                raise ValueError(f"{path}: nonfinite historical record.")
            if "depth" in args.axes:
                check_units(history.scale, args.var, required=True)
                if not np.isfinite(history.scale).all() or (history.scale < 0).any():
                    raise ValueError(f"{path}: invalid historical scale.")
                eligible &= (history.scale > 0).all("month").values
            histories[event], history_paths[event] = history, str(path.resolve())

        cells = np.flatnonzero(eligible.ravel())
        eligible_count = len(cells)
        if args.pilot_cells and len(cells) > args.pilot_cells:
            cells = cells[np.linspace(0, len(cells)-1, args.pilot_cells, dtype=int)]
        if not len(cells):
            raise ValueError("No eligible cells after applying the surface and historical-scale masks.")
        lat, lon = np.unravel_index(cells, lsm.shape)
        index = {"latitude": xr.DataArray(lat, dims="cell"), "longitude": xr.DataArray(lon, dims="cell")}
        weights = np.maximum(np.cos(np.deg2rad(lsm.latitude.values[lat])), 0)
        all_weight = float(len(times) * weights.sum())
        if all_weight <= 0:
            raise ValueError("Zero total area weight.")
        months = xr.DataArray(times, dims="time").dt.month.values - 1
        print(f"Selected cells: {len(cells)} of {eligible_count}; {len(cells)*len(times):,} grid-cell/valid-time cases.", flush=True)
        print("Progress bars count completed Dask read tasks. Scoring progress is printed by lead/event/axis.", flush=True)
        with dask.config.set(scheduler="threads", num_workers=4), ProgressBar():
            observations = np.stack([a.values for a in dask.compute(*[
                truth[r].sel(time=times).isel(index).transpose("time", "cell") for r in references])]).astype(float)
        if not np.isfinite(observations).all():
            raise ValueError("Nonfinite reference values on the common cases; no cases are silently dropped.")

        fingerprint = hashlib.sha256()
        for array in [times, lsm.latitude.values, lsm.longitude.values, cells]:
            fingerprint.update(np.ascontiguousarray(array).tobytes())
        case_fingerprint = fingerprint.hexdigest()
        record_fingerprint = hashlib.sha256()
        for history in histories.values():
            for name in ["record", "scale"] if "depth" in args.axes else ["record"]:
                record_fingerprint.update(np.ascontiguousarray(history[name].values).tobytes())
        units = "K" if args.var == "t2m" else "m/s"

        # 3. Read each forecast lead ONCE, then calculate all four E/T combinations.
        for lead_index, lead in enumerate(args.leads, 1):
            started = time.monotonic()
            print(f"\n[{lead_index}/{len(args.leads)}] {args.var}, {lead} h: reading the four model fields.", flush=True)
            fields = [field_at_lead(sources[m], variable, lead).sel(time=times).isel(index).transpose("time", "cell")
                      for m in args.models]
            with dask.config.set(scheduler="threads", num_workers=4), ProgressBar():
                forecasts = np.stack([a.values for a in dask.compute(*fields)]).astype(float)
            if not np.isfinite(forecasts).all():
                raise ValueError("Nonfinite forecast on the matched cases; no model-specific skipping is allowed.")
            # Physical error retains its sign: cold underprediction of intensity
            # appears as a POSITIVE temperature error (forecast too warm).
            errors = forecasts[None, :, :, :] - observations[:, None, :, :]
            squared_errors = errors**2
            all_bias = (errors * weights).sum(axis=(-2, -1)) / all_weight
            all_mse = (squared_errors * weights).sum(axis=(-2, -1)) / all_weight
            for event, (_, sign) in events.items():
                history = histories[event].isel(index)
                record = history.record.transpose("month", "cell").values[months]
                scale = history.scale.transpose("month", "cell").values[months] if "depth" in args.axes else None
                for axis in args.axes:
                    levels = np.asarray(args.margins if axis == "margin" else args.depths)
                    shape = (2, 2, len(args.models), len(levels))
                    bias, mse = np.full(shape, np.nan), np.full(shape, np.nan)
                    count = np.zeros((2, len(levels)), dtype=np.int64)
                    event_weight = np.zeros((2, len(levels)))
                    for e in range(2):
                        margin = sign * (observations[e] - record)
                        for j, level in enumerate(levels):
                            # Strict cumulative exceedances: D > u or D/s > k.
                            # E alone defines the mask, for every model and BOTH T.
                            mask = margin > (level if axis == "margin" else level * scale)
                            count[e, j] = mask.sum()
                            weighted_mask = mask * weights
                            event_weight[e, j] = weighted_mask.sum()
                            if event_weight[e, j] > 0:
                                bias[e, :, :, j] = (errors * weighted_mask).sum(axis=(-2, -1)) / event_weight[e, j]
                                mse[e, :, :, j] = (squared_errors * weighted_mask).sum(axis=(-2, -1)) / event_weight[e, j]
                    out = xr.Dataset(coords=dict(event_source=references, truth=references,
                                                 model=args.models, level=levels))
                    dims = ["event_source", "truth", "model", "level"]
                    out["bias"] = (dims, bias)
                    out["mse"] = (dims, mse)
                    out["rmse"] = (dims, np.sqrt(mse))
                    out["all_bias"] = (["truth", "model"], all_bias)
                    out["all_mse"] = (["truth", "model"], all_mse)
                    out["all_rmse"] = (["truth", "model"], np.sqrt(all_mse))
                    out["event_count"] = (["event_source", "level"], count)
                    out["event_weight"] = (["event_source", "level"], event_weight)
                    out["event_rate"] = out.event_weight / all_weight
                    if "hres" in args.models:
                        for metric in ["bias", "mse", "rmse"]:
                            out[f"{metric}_difference_from_hres"] = out[metric] - out[metric].sel(model="hres", drop=True)
                        # Same E and same cases => signed between-model bias
                        # differences cannot depend on the verifying reference T.
                        delta = out.bias_difference_from_hres
                        if not np.allclose(delta.sel(truth="ERA5"), delta.sel(truth="HRES_fc0"),
                                           atol=1e-9, rtol=1e-9, equal_nan=True):
                            raise ValueError("Matched-case signed-bias identity failed.")
                    out.attrs = dict(
                        year=args.year, variable=args.var, event=event, threshold_axis=axis,
                        leadtime_hours=lead, surface=args.surface, record_tag=args.record_tag,
                        record_hours_utc=",".join(map(str, args.record_hours)), verification_hours_utc="0,12",
                        case_fingerprint=case_fingerprint, record_fingerprint=record_fingerprint.hexdigest(),
                        common_leads_hours=",".join(map(str, args.leads)), model_names=",".join(args.models),
                        valid_time_count=len(times), selected_cell_count=len(cells), eligible_cell_count=eligible_count,
                        all_count=int(len(times)*len(cells)), all_weight=all_weight,
                        pilot="true" if args.pilot_cells else "false", latitude_domain="-60 < latitude <= 90",
                        records_file=history_paths[event], records_metadata=json.dumps(histories[event].attrs, default=str),
                        reference_files=json.dumps({k: str(v.resolve()) for k, v in reference_paths.items()}),
                        forecast_files=json.dumps({k: str(v.resolve()) for k, v in forecast_paths.items()}),
                        source_urls=json.dumps({k: str(v.attrs.get("source_url", "not recorded")) for k, v in sources.items()}),
                        event_definition="sign*(E-R)>level for margin; sign*(E-R)>level*s for depth; sign=-1 for cold",
                        bias_definition="forecast minus T, in physical units; positive cold bias means too warm",
                        aggregation="Cosine-latitude weighted mean of errors/squared errors; RMSE=sqrt(weighted MSE)",
                        interpretation="Outcome-conditioned diagnostic of raw deterministic forecasts; no EasyUQ fit; no cap identification")
                    for metric in ["bias", "rmse", "all_bias", "all_rmse", "bias_difference_from_hres", "rmse_difference_from_hres"]:
                        if metric in out:
                            out[metric].attrs["units"] = units
                    out.mse.attrs["units"] = out.all_mse.attrs["units"] = f"({units})^2"
                    out.level.attrs = dict(units=units if axis == "margin" else "1",
                                           description="Chosen cumulative exceedance threshold, not a disjoint bin")
                    stem = f"reference_{event}_{axis}_{args.year}_{args.record_tag}_{args.surface}_{lead:03d}h"
                    # Write the compact table before adding independent time/cell
                    # coordinates, avoiding an accidental huge Cartesian CSV.
                    table = out.to_dataframe().reset_index()
                    for key in ["year", "variable", "event", "threshold_axis", "leadtime_hours", "surface", "record_tag", "pilot"]:
                        table[key] = out.attrs[key]
                    table.to_csv(args.output / f"{stem}.csv.partial", index=False)
                    out = out.assign_coords(verification_time=times, cell=cells)
                    out["cell_latitude"] = ("cell", lsm.latitude.values[lat])
                    out["cell_longitude"] = ("cell", lsm.longitude.values[lon])
                    out["cell_weight"] = ("cell", weights)
                    out.to_netcdf(args.output / f"{stem}.nc.partial", engine="netcdf4")
                    os.replace(args.output / f"{stem}.nc.partial", args.output / f"{stem}.nc")
                    os.replace(args.output / f"{stem}.csv.partial", args.output / f"{stem}.csv")
                    print(f"  Saved {stem}; event counts E=ERA5 {count[0].tolist()}, E=HRES_fc0 {count[1].tolist()}", flush=True)
            print(f"Lead completed in {(time.monotonic()-started)/60:.1f} min.", flush=True)
    print(f"Done. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
