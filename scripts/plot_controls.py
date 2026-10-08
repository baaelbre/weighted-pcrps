"""Sequential plots of the EasyUQ controls and paired contribution tables."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path("results/controls"))
parser.add_argument("--output", type=Path, default=Path("figures/controls"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
paths = sorted(args.results.glob("controls_*.nc"))
if not paths:
    raise FileNotFoundError("No control summaries found.")
rows = []
for path in paths:
    with xr.open_dataset(path) as source:
        ds = source.squeeze("leadtime", drop=True).load()
        lead = int(source.leadtime.values[0])
    for event in ds.event.values:
        data = ds.sel(event=event)
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), layout="constrained")
        panels = [("squared_error", "all", "RMSE: all cases"), ("squared_error", "record", "RMSE: observed exceedances"),
                  ("crps", "all", "CRPS: all cases"), ("crps", "record", "CRPS: observed exceedances"),
                  ("rtwcrps", "all", "rtwCRPS: all cases"), ("brier", "all", "Brier: all cases")]
        for ax, (score, subset, title) in zip(axes.flat, panels):
            for j, model in enumerate(ds.model.values):
                # Raw vs full EasyUQ makes the central comparison readable.
                # Centre controls are shown separately below and retained in the CSV.
                for rep, style in [("raw", "--"), ("easyuq", "-")]:
                    values = data[subset].sel(model=model, representation=rep, score=score)
                    if score == "squared_error":
                        values = np.sqrt(values)
                    ax.plot(data.depth, values, color=f"C{j}", linestyle=style, marker="o", label=f"{model}: {rep}")
            unit = "K" if ds.attrs["variable"] == "t2m" else "m/s"
            ax.set(title=title, xlabel="Threshold depth k", ylabel="Score" if score == "brier" else unit)
            ax.set_xticks(data.depth.values)
            if data.sizes["depth"] > 1:
                ax.set_xlim(float(data.depth.min()), float(data.depth.max()))
            ax.grid(True, alpha=0.4)
        axes.flat[0].legend(fontsize=7)
        pilot = "; pilot" if ds.attrs.get("pilot") == "true" else ""
        fig.suptitle(f"{event.capitalize()} records, {lead} h, {ds.attrs['surface']}\nEasyUQ {ds.attrs['mode']}; {ds.attrs['selected_cell_count']} cells{pilot}")
        stem = path.stem + "_" + str(event)
        fig.savefig(args.output / f"{stem}.png", dpi=160)
        fig.savefig(args.output / f"{stem}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for model, ax in zip(ds.model.values, axes.flat):
            for rep in ds.representation.values:
                ax.plot(data.depth, data.record.sel(model=model, representation=rep, score="crps"), "o-", label=rep)
            ax.set(title=str(model), xlabel="Threshold depth k", ylabel="Record-conditioned CRPS")
            ax.grid(True, alpha=0.4)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Distribution versus its centres: {event}, {lead} h ({ds.attrs['mode']})")
        fig.savefig(args.output / f"{stem}_centres.png", dpi=160)
        plt.close(fig)

        # Like-for-like model rankings and within-model mechanisms. Every delta
        # is model - reference: negative favours the first entry.
        comparisons = [(model, rep, "hres", rep) for model in ds.model.values if model != "hres"
                       for rep in ds.representation.values] if "hres" in ds.model.values else []
        comparisons += [(model, a, model, b) for model in ds.model.values for a, b in
                        [("easyuq_mean", "raw"), ("easyuq_median", "raw"), ("easyuq", "easyuq_mean"), ("easyuq", "easyuq_median"), ("easyuq", "raw")]]
        for model, rep, ref_model, ref_rep in comparisons:
            a = data.sel(model=model, representation=rep)
            b = data.sel(model=ref_model, representation=ref_rep)
            for score in ["crps", "rtwcrps", "brier", "bias", "squared_error"]:
                for k in data.depth.values:
                    values = {subset: float((a[subset]-b[subset]).sel(score=score, depth=k))
                              for subset in ["all", "record", "nonrecord", "record_contribution", "nonrecord_contribution"]}
                    if not np.isclose(values["all"], values["record_contribution"]+values["nonrecord_contribution"], atol=1e-10):
                        raise ValueError("Contribution identity failed.")
                    rows.append(dict(file=path.name, event=event, leadtime=lead, mode=ds.attrs["mode"], depth=float(k),
                                     model=model, representation=rep, reference_model=ref_model, reference_representation=ref_rep,
                                     score=score, record_count=int(data.record_count.sel(depth=k)), **values))
pd.DataFrame(rows).to_csv(args.output / "paired_differences.csv", index=False)
print(f"Saved figures and paired_differences.csv in {args.output}.")
