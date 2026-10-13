"""Sequential Matplotlib plots for the remaining raw 2020 record diagnostics.

Edit colours/figsizes here. No IDR fit, forecast loading, or uncertainty calculation.
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
parser.add_argument("--results", type=Path, default=Path("results/reference_occurrence"))
parser.add_argument("--output", type=Path, default=Path("figures/reference_occurrence"))
parser.add_argument("--conditional-results", type=Path, default=Path("results/reference_comparison"),
                    help="When matching old conditional files exist, check their dates/grid/records against this run.")
args = parser.parse_args()
paths = sorted(args.results.glob("occurrence_*.nc"))
if not paths:
    raise FileNotFoundError(f"No occurrence_*.nc summaries in {args.results}.")
args.output.mkdir(parents=True, exist_ok=True)
colours = {"hres": "black", "graphcast": "tab:blue", "pangu": "tab:orange", "fuxi": "tab:green",
           "ERA5": "tab:purple", "HRES_fc0": "0.5"}
labels = {"hres": "HRES", "graphcast": "GraphCast", "pangu": "Pangu", "fuxi": "FuXi",
          "ERA5": "ERA5", "HRES_fc0": "HRES-fc0"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
pr_tables, count_tables, correlation_tables, joint_tables, agreement_rows = [], [], [], [], []
signatures, reference_counts = {}, {}

for path in paths:
    with xr.open_dataset(path) as source:
        ds = source.drop_vars(["monthly_forecast_sd", "cell_latitude", "cell_longitude", "cell_weight"]).load()
    a = ds.attrs
    event, variable, lead = a["event"], a["variable"], int(a["leadtime_hours"])
    tag, year, surface = a["record_tag"], int(a["year"]), a["surface"]
    pilot = " [PILOT]" if a["pilot"] == "true" else ""
    title = f"{event.capitalize()} records (ERA5 {tag.replace('_', '–')}, {surface}, {year}, {lead} h){pilot}"
    key = (variable, tag, surface)
    signature = tuple(a[k] for k in ["case_fingerprint", "record_fingerprint", "model_names", "record_hours_utc", "pilot"])
    signature += (tuple(ds.gain.values), a["forecast_sd_ddof"])
    if key in signatures and signatures[key] != signature:
        raise ValueError("Different cases, records or settings in this folder. Keep different runs separate.")
    signatures[key] = signature
    old = args.conditional_results / f"reference_{event}_depth_{year}_{tag}_{surface}_{lead:03d}h.nc"
    if old.exists():
        with xr.open_dataset(old) as previous:
            for name in ["case_fingerprint", "record_fingerprint", "model_names", "record_hours_utc"]:
                if previous.attrs[name] != a[name]:
                    raise ValueError(f"{old.name}: {name} differs from the new occurrence run.")
            if 0 in previous.level:
                np.testing.assert_array_equal(previous.event_count.sel(level=0), ds.observed_count)

    # 1. PR curves: one common reference per column. Count and area weighting
    # are both shown. The dot is gain=0, and the dashed line is event prevalence.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for row, weighting in enumerate(["count", "area"]):
        for col, truth in enumerate(ds.truth.values):
            ax = axes[row, col]
            for model in ds.model.values:
                s = ds.sel(weighting=weighting, truth=truth, model=model)
                ax.plot(s.recall, s.precision, color=colours[str(model)], label=labels[str(model)])
                ax.plot(float(s.recall.sel(gain=0)), float(s.precision.sel(gain=0)), "o", color=colours[str(model)])
            prevalence = float(ds.observed_rate.sel(weighting=weighting, truth=truth).isel(model=0, gain=0))
            ax.axhline(prevalence, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_title(f"{labels[str(truth)]}; {weighting}; observed {100*prevalence:.3f}%")
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.grid(True, alpha=0.4)
            if row == 1:
                ax.set_xlabel("Recall")
            if col == 0:
                ax.set_ylabel("Precision")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(title + f"\nPR: gain {float(ds.gain.min()):g} to {float(ds.gain.max()):g}; dots = raw forecasts", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"{path.stem}_pr.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Raw confusion tables. Numbers are counts; shading and percentages
    # normalize each observed row, so the many true negatives do not hide hits.
    fig, axes = plt.subplots(2, len(ds.model), figsize=(3.2*len(ds.model), 5.8), squeeze=False, layout="constrained")
    for row, truth in enumerate(ds.truth.values):
        for col, model in enumerate(ds.model.values):
            ax = axes[row, col]
            tp, fp, fn, tn = ds.confusion_count.sel(truth=truth, model=model, gain=0).values
            matrix = np.array([[tn, fp], [fn, tp]])
            denominator = matrix.sum(axis=1, keepdims=True)
            rates = np.divide(matrix, denominator, out=np.full((2, 2), np.nan), where=denominator > 0)
            ax.imshow(np.ma.masked_invalid(rates), vmin=0, vmax=1, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    pct = f"{100*rates[i,j]:.1f}%" if np.isfinite(rates[i,j]) else "undefined"
                    ax.text(j, i, f"{matrix[i,j]:,}\n{pct}", ha="center", va="center", fontsize=9,
                            color="white" if rates[i,j] > 0.55 else "black")
            ax.set_title(f"{labels[str(model)]} | {labels[str(truth)]}")
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_yticks([0, 1], ["No", "Yes"])
            ax.set_xlabel("Forecast record")
            if col == 0:
                ax.set_ylabel("Observed record")
    fig.suptitle(title + "\nRaw forecasts: counts and observed-row percentages", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"{path.stem}_confusion.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3. Binary indicator correlations, including ERA5 versus HRES-fc0.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, weighting in zip(axes, ["count", "area"]):
        values = ds.indicator_correlation.sel(weighting=weighting).values
        im = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(len(ds.source)):
            for j in range(len(ds.source)):
                text = f"{values[i,j]:.2f}" if np.isfinite(values[i,j]) else "—"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="white" if abs(values[i,j]) > 0.65 else "black")
        ax.set_xticks(range(len(ds.source)), [labels[str(s)] for s in ds.source.values], rotation=45, ha="right")
        ax.set_yticks(range(len(ds.source)), [labels[str(s)] for s in ds.source.values])
        ax.set_title(f"{weighting} weighting")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Binary correlation (phi)")
    fig.suptitle(title + "\nRecord indicators at gain=0", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"{path.stem}_correlation.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Compact tables only: do NOT include independent cell/time/SD dimensions.
    metrics = ["precision", "recall", "false_positive_rate", "f1", "phi", "observed_rate", "forecast_rate"]
    table_ds = ds[metrics + ["observed_count", "observed_weight", "forecast_count", "forecast_weight"]].copy()
    for outcome in ["TP", "FP", "FN", "TN"]:
        table_ds[f"{outcome.lower()}_count"] = ds.confusion_count.sel(outcome=outcome, drop=True)
        table_ds[f"{outcome.lower()}_weight"] = ds.confusion_weight.sel(outcome=outcome, drop=True)
    frame = table_ds.to_dataframe().reset_index()
    count_frame = ds[["source_count", "source_weight"]].to_dataframe().reset_index()
    count_frame["rate_count"] = count_frame.source_count / a["all_count"]
    count_frame["rate_area"] = count_frame.source_weight / a["all_weight"]
    corr_frame = ds[["indicator_correlation", "joint_exceedance_count", "joint_exceedance_weight"]].to_dataframe().reset_index()
    tables = [frame, count_frame, corr_frame]
    if "joint_forecast_rmse" in ds:
        names = [n for n in ds.data_vars if n.startswith("joint_forecast_")]
        joint_frame = ds[names].to_dataframe().reset_index()
        tables.append(joint_frame)
    for table in tables:
        for name in ["event", "variable", "year", "record_tag", "surface", "leadtime_hours", "pilot", "case_fingerprint"]:
            table[name] = a[name]
    pr_tables.append(frame)
    count_tables.append(count_frame)
    correlation_tables.append(corr_frame)
    if "joint_forecast_rmse" in ds:
        joint_tables.append(joint_frame)

    # ERA5/HRES-fc0 agreement is independent of the forecast lead.
    key = (event, year, tag, surface)
    values = np.r_[ds.source_count.sel(source=["ERA5", "HRES_fc0"]).values,
                   ds.joint_exceedance_count.sel(source="ERA5", other_source="HRES_fc0").item()]
    if key in reference_counts:
        np.testing.assert_array_equal(reference_counts[key], values)
    else:
        reference_counts[key] = values
        for weighting, suffix, total in [("count", "count", a["all_count"]), ("area", "weight", a["all_weight"])]:
            era = float(ds[f"source_{suffix}"].sel(source="ERA5"))
            hres = float(ds[f"source_{suffix}"].sel(source="HRES_fc0"))
            both = float(ds[f"joint_exceedance_{suffix}"].sel(source="ERA5", other_source="HRES_fc0"))
            agreement_rows.append(dict(event=event, year=year, record_tag=tag, surface=surface, weighting=weighting,
                                       both=both, era5_only=era-both, hres_fc0_only=hres-both,
                                       neither=total-era-hres+both, total=total,
                                       correlation=float(ds.indicator_correlation.sel(weighting=weighting, source="ERA5", other_source="HRES_fc0"))))
    print(f"Plotted {path.name}", flush=True)

pr = pd.concat(pr_tables, ignore_index=True)
counts = pd.concat(count_tables, ignore_index=True)
pr.to_csv(args.output / "precision_recall.csv", index=False)
pr[pr.gain == 0].to_csv(args.output / "occurrence_summary.csv", index=False)
counts.to_csv(args.output / "record_counts.csv", index=False)
pd.concat(correlation_tables, ignore_index=True).to_csv(args.output / "indicator_correlations.csv", index=False)
pd.DataFrame(agreement_rows).to_csv(args.output / "reference_agreement.csv", index=False)

# 4. Predicted and observed record frequencies versus lead time.
for (event, year, tag, surface), frame in counts.groupby(["event", "year", "record_tag", "surface"]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, series in frame.groupby("source", sort=False):
        series = series.sort_values("leadtime_hours")
        style = "--" if name in ["ERA5", "HRES_fc0"] else "o-"
        axes[0].plot(series.leadtime_hours, series.source_count, style, color=colours[name], label=labels[name])
        axes[1].plot(series.leadtime_hours, 100*series.rate_area, style, color=colours[name], label=labels[name])
    axes[0].set_ylabel("Record grid-cell / time pairs")
    axes[1].set_ylabel("Area-weighted occurrence (%)")
    for ax in axes:
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks(sorted(frame.leadtime_hours.unique()))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.4)
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    pilot = " [PILOT]" if (frame.pilot == "true").any() else ""
    fig.suptitle(f"{event.capitalize()} records ({year}, ERA5 {tag.replace('_', '–')}, {surface}){pilot}")
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"counts_{event}_{year}_{tag}_{surface}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

# 5. Jointly forecast records: one figure per AI/HRES pair, common subset for
# both systems. Area-weighted errors; raw case counts are given below the title.
if joint_tables:
    joint = pd.concat(joint_tables, ignore_index=True)
    joint.to_csv(args.output / "joint_forecast_scores.csv", index=False)
    for (event, variable, year, tag, surface, model), frame in joint[joint.weighting == "area"].groupby(
            ["event", "variable", "year", "record_tag", "surface", "comparison_model"]):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row", layout="constrained")
        units = "K" if variable == "t2m" else "m/s"
        for col, truth in enumerate(["ERA5", "HRES_fc0"]):
            for role, name in [("ai", model), ("hres", "hres")]:
                series = frame[(frame.truth == truth) & (frame.forecast_role == role)].sort_values("leadtime_hours")
                axes[0, col].plot(series.leadtime_hours, series.joint_forecast_rmse, "o-", color=colours[name], label=labels[name])
                axes[1, col].plot(series.leadtime_hours, series.joint_forecast_bias, "o-", color=colours[name], label=labels[name])
            axes[0, col].set_title(f"T: {labels[truth]}")
            axes[1, col].set_xlabel("Lead time (h)")
            axes[1, col].axhline(0, color="0.5", linestyle="--", linewidth=0.8)
            for row in range(2):
                axes[row, col].set_xticks(sorted(frame.leadtime_hours.unique()))
                axes[row, col].grid(True, alpha=0.4)
        finite = frame.joint_forecast_rmse.dropna()
        axes[0, 0].set_ylim(0, max(float(finite.max())*1.08, 0.01) if len(finite) else 1)
        axes[0, 0].set_ylabel(f"RMSE ({units})")
        axes[1, 0].set_ylabel(f"Mean error ({units})")
        axes[0, 0].legend(frameon=False)
        sizes = frame[["leadtime_hours", "joint_forecast_count"]].drop_duplicates().sort_values("leadtime_hours")
        count_text = "; ".join(f"{int(r.leadtime_hours)} h: n={int(r.joint_forecast_count)}" for r in sizes.itertuples())
        pilot = " [PILOT]" if (frame.pilot == "true").any() else ""
        fig.suptitle(f"{event.capitalize()}: both {labels[model]} and HRES forecast a record ({year}, {surface}){pilot}\n{count_text}", fontsize=12)
        for ext in ["png", "pdf"]:
            fig.savefig(args.output / f"joint_forecast_{event}_{model}_{year}_{tag}_{surface}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)
print(f"Done. Figures and CSV tables: {args.output}", flush=True)
