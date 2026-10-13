# Raw forecasts: event selection and verification reference

This experiment holds the forecast fields fixed and evaluates all four models
in every cell of this table. It adds to the EasyUQ work without fitting IDR again.

The remaining occurrence and jointly forecast record experiments are now available
in [REFERENCE_OCCURRENCE.md](REFERENCE_OCCURRENCE.md), with their own runner.

| Event selection E | Verification T = ERA5 | Verification T = HRES-fc0 |
| --- | --- | --- |
| ERA5 | Common ERA5 comparison | Same ERA5 events, changed verification |
| HRES-fc0 | Same HRES-fc0 events, changed verification | Common HRES-fc0 comparison |

E selects the cases. T supplies the value subtracted from the forecast. HRES is
a forecast model; HRES-fc0 is a verification reference. ERA5 historical thresholds
remain fixed for every model and every combination. Exceedances of these thresholds
by HRES-fc0 are not records relative to a separate long HRES-fc0 history.

## What is calculated

- Models: HRES, GraphCast, Pangu and FuXi, raw deterministic forecasts.
- Verification: 2020, 00/12 UTC, land, excluding latitude <= -60 degrees.
- Default leads: 48, 120 and 240 hours.
- Historical monthly records: ERA5 1979–2019 at 00/12 UTC.
- Heat/wind margin: D = E - R. Cold margin: D = R - E.
- Physical-margin curves select D > u, with u = 0, 0.5, 1, 2, 3, 4, 5, 6 K or m/s.
- Standardized-depth curves select D > k*s, with k = 0, 0.5, 1, 2 and the existing
  local sample SD across yearly monthly extremes. These are cumulative selections,
  not disjoint bins. With both axes requested, cells need positive scales in all
  months for all events of that variable; the same cells enter every comparison.
- Cosine-weighted bias, MSE and RMSE on all cases and selected exceedances.
- Event counts, weighted rates and paired model-minus-HRES differences.

Bias always means forecast minus T in physical units: a positive cold bias means
too warm. RMSE is the square root of the weighted mean squared error. The same
dates are used across BOTH references, every model and every requested lead.
This may omit early January and source gaps. Changing the requested lead/model
list can change that common set; keep different runs in separate output folders.
Missing values or mismatched grids cause an error, not per-model case dropping.

The conditional curves are descriptive diagnostics. They are not proper
all-case probabilistic scores or an identification test for a soft cap. The
existing EasyUQ controls supply CRPS/twCRPS comparisons separately; their saved
summaries cannot be truth-switched without the underlying forecasts/observations.
This new evaluator does not make a fair-score correction to a deterministic point.

## Biobot: the data are already present

Use the files listed in [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md), under
`/data/nvme1/bastiaan/probex-data`. No copying, downloads, GPU, compiler or new
dependency is needed in the existing `probex-easyuq` environment.

After incorporating this package into the repository and pulling it on Biobot:

```bash
cd /home/bastiaan/weighted-pcrps
git pull --ff-only
export PCRPS_PYTHON=/opt/miniconda3/envs/probex-easyuq/bin/python
export PCRPS_DATA_DIR=/data/nvme1/bastiaan/probex-data
export PCRPS_RECORDS_ROOT="$PCRPS_DATA_DIR/era5_records"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

### Small execution check

This uses 32 spread-out land cells and all common dates. Empty tails are normal
in this small sample. It checks the pipeline, not model rankings.

```bash
"$PCRPS_PYTHON" -u -m pcrps.compute_reference_comparison \
    --var t2m --leads 48 --pilot-cells 32 \
    --output results/reference_pilot
"$PCRPS_PYTHON" scripts/plot_reference_comparison.py \
    --results results/reference_pilot --output figures/reference_pilot
```

### Full experiment, including figures

```bash
mkdir -p logs
PCRPS_REFERENCE_LOG="logs/reference_comparison_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash jobs/run_reference_comparison.sh \
    > "$PCRPS_REFERENCE_LOG" 2>&1 < /dev/null &
echo "Started PID: $!"
echo "Log: $PCRPS_REFERENCE_LOG"
tail -f "$PCRPS_REFERENCE_LOG"
```

Ctrl+C stops `tail`, not the background calculation. Reads use four Dask threads;
variables and leads are processed sequentially. Each lead is read once and used
for all four E/T combinations. Progress bars count read tasks; scoring prints
each completed event/axis. Use the first complete lead's elapsed time as the
runtime guide; download speed or the old IDR runtime is not a reliable estimate.

The run saves small NetCDF/CSV files in `results/reference_comparison/` and PNG/PDF
figures in `figures/reference_comparison/`. Each NetCDF retains exact verification
dates, selected cell coordinates, weights, baseline metadata and source paths.
Rerunning the same command replaces its summaries; it does not modify any input
data or the existing EasyUQ results. No bootstrap or uncertainty calculation is run.

## Reading the figures

Each conditional bias/RMSE figure has four panels: rows change E, columns change
T. Read across a row to isolate the reference change at fixed events. Read down
a column to see how event selection changes the result at a fixed reference.

The signed model-minus-HRES bias gap must be identical across T at fixed E and
threshold. The evaluator checks this identity. Absolute bias and RMSE rankings
can still change. Empty event sets have NaN scores, not zero scores. Counts are
grid-cell/time pairs, not independent storms; their plots use a logarithmic count
axis above one and show zero explicitly. All-case figures show the wider context.

The plotter also writes `reference_comparison.csv` and `all_case_scores.csv`.
Negative MSE/RMSE differences favour the named model over HRES. A negative signed
bias difference does not automatically mean a better forecast; compare bias to zero.

## A closer Zhang sensitivity (optional)

The default uses the records you already calculated. It is a harmonized 1.5-degree
sensitivity analysis, not an exact reproduction of Zhang's 0.25-degree experiment,
whose records use 1979–2017 and four historical times per day. Spatial averaging,
record period and temporal sampling can all affect the comparison.

If you additionally build 1979–2017 record files using 00/06/12/18 UTC, point to
those files with `--records-root`, use `--record-tag 1979_2017 --record-hours 0 6 12 18`,
and choose a separate output folder. Use `--axes margin` if those files contain no
scales, or compute matching scales to retain both axes. The evaluator checks the
historical metadata instead of relabelling the existing 2019 records. Verification
still uses 2020 at 00/12 UTC. Do not mix runs with different records/hours in one
results folder. The original EasyUQ and paper runners remain unchanged apart from
deferring IDR imports until the EasyUQ entry point is actually used.
