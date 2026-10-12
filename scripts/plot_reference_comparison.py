"""Simple Matplotlib figures for the raw-forecast reference comparison.

Rows of each 2x2 figure change event source E. Columns change verifying truth T.
Plotting is sequential; edit the colours, labels or figsize directly below.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path("results/reference_comparison"))
parser.add_argument("--output", type=Path, default=Path("figures/reference_comparison"))
args = parser.parse_args()
paths = sorted(args.results.glob("reference_*.nc"))
if not paths:
    raise FileNotFoundError(f"No reference_*.nc summaries in {args.results}.")
args.output.mkdir(parents=True, exist_ok=True)

colours = {"hres": "black", "graphcast": "tab:blue", "pangu": "tab:orange", "fuxi": "tab:green"}
labels = {"hres": "HRES", "graphcast": "GraphCast", "pangu": "Pangu", "fuxi": "FuXi",
          "ERA5": "ERA5", "HRES_fc0": "HRES-fc0"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11})
overall_rows, score_rows = [], []
case_checks, count_checks = {}, {}

for path in paths:
    with xr.open_dataset(path) as source:
        ds = source.load()
    a = ds.attrs
    variable, event, axis = a["variable"], a["event"], a["threshold_axis"]
    lead = int(a["leadtime_hours"])
    units = "K" if variable == "t2m" else "m/s"
    pilot = " [PILOT]" if a["pilot"] == "true" else ""
    title = f"{event.capitalize()} records (ERA5 {a['record_tag'].replace('_', '–')}, {a['surface']}, {lead} h){pilot}"
    xlabel = f"Minimum observed margin ({units})" if axis == "margin" else "Minimum observed depth k"
    key = (variable, a["record_tag"], a["surface"])
    signature = (a["case_fingerprint"], a["record_fingerprint"], a["model_names"], a["record_hours_utc"], a["pilot"])
    if key in case_checks and case_checks[key] != signature:
        raise ValueError("These summaries do not share cases/records/models. Put different runs in separate results folders.")
    case_checks[key] = signature

    # 1. Two four-panel figures: conditional RMSE and signed conditional bias.
    # Empty tails remain NaN and are not plotted as a zero score.
    for metric, ylabel in [("rmse", f"RMSE ({units})"), ("bias", f"Mean error ({units})")]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.84, bottom=0.11, left=0.10, right=0.98, hspace=0.30, wspace=0.12)
        for row, event_source in enumerate(ds.event_source.values):
            for col, truth in enumerate(ds.truth.values):
                ax = axes[row, col]
                for model in ds.model.values:
                    values = ds[metric].sel(event_source=event_source, truth=truth, model=model)
                    ax.plot(ds.level, values, "o-", color=colours[str(model)], label=labels[str(model)], markersize=4)
                ax.set_title(f"E: {labels[str(event_source)]}  |  T: {labels[str(truth)]}")
                ax.set_xticks(ds.level.values)
                if len(ds.level) > 1:
                    ax.set_xlim(float(ds.level.min()), float(ds.level.max()))
                if row == 1:
                    ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if metric == "bias":
                    ax.axhline(0, color="0.45", linewidth=0.8, linestyle="--")
                ax.grid(True, alpha=0.4)
        # Set limits AFTER every panel is drawn: setting one shared limit early
        # would freeze the upper bound and could clip another reference's curve.
        if metric == "rmse":
            finite = ds[metric].values[np.isfinite(ds[metric].values)]
            upper = max(float(finite.max())*1.08, 0.01) if finite.size else 1.0
            axes[0, 0].set_ylim(0, upper)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=len(names), frameon=False)
        fig.suptitle(title, fontsize=14, y=0.985)
        for extension in ["png", "pdf"]:
            fig.savefig(args.output / f"{path.stem}_{metric}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # 2. Counts and area-weighted rates. They depend on E, not on model or T,
    # and are identical across leads because we used a common valid-time set.
    count_key = (*key, event, axis)
    counts = ds.event_count.values
    if count_key in count_checks:
        old_levels, old_counts = count_checks[count_key]
        if not np.array_equal(old_levels, ds.level) or not np.array_equal(old_counts, counts):
            raise ValueError("Event levels/counts changed between leads despite a common case set.")
    else:
        count_checks[count_key] = (ds.level.values, counts)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for event_source, colour in [("ERA5", "tab:blue"), ("HRES_fc0", "black")]:
            axes[0].plot(ds.level, ds.event_count.sel(event_source=event_source), "o-", color=colour, label=labels[event_source])
            axes[1].plot(ds.level, 100*ds.event_rate.sel(event_source=event_source), "o-", color=colour, label=labels[event_source])
        axes[0].set_ylabel("Grid-cell / time pairs")
        axes[0].set_yscale("symlog", linthresh=1)
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylabel("Area-weighted event rate (%)")
        axes[1].set_ylim(bottom=0)
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.set_xticks(ds.level.values)
            ax.grid(True, alpha=0.4)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{event.capitalize()} records: event selection ({a['surface']}, {a['year']}){pilot}")
        count_stem = f"counts_{event}_{axis}_{a['year']}_{a['record_tag']}_{a['surface']}"
        for extension in ["png", "pdf"]:
            fig.savefig(args.output / f"{count_stem}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Only score dimensions enter these tables, not the independent saved
    # verification_time/cell coordinates.
    names = ["bias", "mse", "rmse", "event_count", "event_weight", "event_rate", "all_bias", "all_mse", "all_rmse"]
    names += [name for name in ds.data_vars if name.endswith("_difference_from_hres")]
    frame = ds[names].to_dataframe().reset_index()
    for name in ["year", "variable", "event", "threshold_axis", "leadtime_hours", "surface", "record_tag", "pilot"]:
        frame[name] = a[name]
    score_rows.append(frame)
    frame = ds[["all_bias", "all_mse", "all_rmse"]].to_dataframe().reset_index()
    for name in ["year", "variable", "leadtime_hours", "surface", "record_tag", "pilot"]:
        frame[name] = a[name]
    overall_rows.append(frame)
    print(f"Plotted {path.name}", flush=True)

# 3. Overall performance versus lead, with a common reference in each column.
overall = pd.concat(overall_rows, ignore_index=True).drop_duplicates()
for (variable, year, tag, surface), frame in overall.groupby(["variable", "year", "record_tag", "surface"]):
    units = "K" if variable == "t2m" else "m/s"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.10, right=0.98, hspace=0.30, wspace=0.12)
    for col, truth in enumerate(["ERA5", "HRES_fc0"]):
        for model, series in frame[frame.truth == truth].groupby("model", sort=False):
            series = series.sort_values("leadtime_hours")
            for row, metric in enumerate(["all_rmse", "all_bias"]):
                axes[row, col].plot(series.leadtime_hours, series[metric], "o-", color=colours[model], label=labels[model])
        axes[0, col].set_title(f"T: {labels[truth]}")
        axes[1, col].axhline(0, color="0.45", linestyle="--", linewidth=0.8)
        axes[1, col].set_xlabel("Lead time (h)")
        for row in range(2):
            axes[row, col].set_xticks(sorted(frame.leadtime_hours.unique()))
            axes[row, col].grid(True, alpha=0.4)
    axes[0, 0].set_ylabel(f"RMSE ({units})")
    axes[0, 0].set_ylim(0, max(float(frame.all_rmse.max())*1.08, 0.01))
    axes[1, 0].set_ylabel(f"Mean error ({units})")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=len(names), frameon=False)
    pilot = " [PILOT]" if (frame.pilot == "true").any() else ""
    fig.suptitle(f"{variable}: all cases ({year}, {surface}){pilot}", fontsize=14, y=0.985)
    for extension in ["png", "pdf"]:
        fig.savefig(args.output / f"all_{variable}_{year}_{tag}_{surface}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
overall.to_csv(args.output / "all_case_scores.csv", index=False)
pd.concat(score_rows, ignore_index=True).to_csv(args.output / "reference_comparison.csv", index=False)
print(f"Done. Figures and combined CSV tables: {args.output}")
