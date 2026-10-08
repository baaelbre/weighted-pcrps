# EasyUQ control experiment

This addition leaves the original paper's Beam runners, helper formulas and
environments unchanged. New entry point: `python -m pcrps.compute_controls`.

For **HRES, GraphCast, Pangu and FuXi**, it fits IDR once per grid cell/model/lead
and compares raw point forecasts, the EasyUQ mean as a point mass, its median as a
point mass, and the full EasyUQ distribution. IDR is fitted across the available
dates at each cell, as in the existing helper; historical thresholds still depend
on calendar month. It does not fit a separate sparse IDR to realized records.

| Comparison, using identical cases | What it isolates |
|---|---|
| Raw point vs EasyUQ mean, same bias/MSE | Change in the forecast centre after recalibration |
| Raw point vs EasyUQ median, same absolute error/point CRPS | Recalibrated median |
| EasyUQ distribution vs its mean or median, same CRPS | Distribution versus a collapsed point |
| Same distribution: all-case vs record-conditioned CRPS | Outcome selection |
| Same distribution: all-case CRPS vs all-case rtwCRPS | Threshold weighting |
| HRES vs each AI system, same representation | Like-for-like model comparison |

All use common valid dates across models and requested leads, one reference,
fixed historical records/scales, common land mask and cosine latitude weights.
The evaluator rejects grid mismatch and missing values rather than silently
changing cases. It uses precomputed WB2 wind speed, not speed from averaged
vectors. Cells with nonpositive monthly scales are excluded consistently.

EasyUQ support points have probability masses. They are **not ensemble samples**:
the scores are exact weighted-distribution CRPS/rtwCRPS/Brier, with no fair
correction and no Monte Carlo approximation. Upper and lower tails are included.
The output keeps conditional means, event counts, probabilities and the paired
decomposition `all = record_contribution + nonrecord_contribution`.
For squared error take the square root of the mean to obtain RMSE; never add RMSEs.

## Interpretation boundary

Default `--mode in-sample` reproduces the **potential-information** approach:
IDR sees the observations being scored. These results are not issued-forecast
skill, nor a test of forecasting values absent from its calibration data.
Monotone amplitude compression can preserve its predictor ordering and hence
its in-sample fit. Native ensembles in ProbEx answer a different question.

The optional `--mode holdout --calibration-end ...` uses only earlier valid dates
for IDR. Evaluation is strictly after that date. This is a within-year chronological
holdout, not a guarantee that the underlying AI checkpoint never used that year.
Basic IDR has support on its calibration observations. It cannot establish
unrestricted tail extrapolation. Model checkpoint provenance still matters.

For the deterministic bridge use 2020, which has public output for all four
models, and thresholds/scales from **1979–2019**. Using 1979–2021 records to
evaluate 2020 would leak future information. This is a controlled extension of
the existing experiment, not an exact numerical reproduction of all its original
variables, lead times, dates and masks. Do not merge these rows into 2022 results.

## Biobot setup

Use a clone of the existing environment so the new IDR dependency does not
alter the working ProbEx environment. The pinned IDR package needs a C++ compiler
and `setuptools<81`; the supplied requirements include the compatible pin.

```bash
cd /home/bastiaan/weighted-pcrps
conda create -n probex-easyuq --clone probex
conda activate probex-easyuq
python -m pip install -r requirements-controls.txt
export PCRPS_PYTHON=/opt/miniconda3/envs/probex-easyuq/bin/python
export PCRPS_DATA_DIR=/data/nvme1/bastiaan/probex-data/easyuq_2020
export PCRPS_RECORDS_ROOT=/data/nvme1/bastiaan/probex-data/era5_records
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

If the compiler is missing, install it in this environment with
`conda install -c conda-forge cxx-compiler`, then retry pip. No GPU is needed.

### Prepare missing inputs once

Skip the forecast download if the expected local files already exist with all
requested leads. The new downloader verifies existing metadata and keeps those
files. It downloads two surface variables and three leads on 1.5 degrees; it
does not generate forecasts or download the full-resolution archives.

```bash
"$PCRPS_PYTHON" -u scripts/prepare_control_data.py \
    --data-dir "$PCRPS_DATA_DIR" --leads 48 120 240

cd /home/bastiaan/probex
/opt/miniconda3/envs/probex/bin/python -u download/create_era5_records.py \
    --start-year 1979 --end-year 2019 --record-tag 1979_2019 \
    --out-root "$PCRPS_RECORDS_ROOT" --with-scale
cd /home/bastiaan/weighted-pcrps
```

The existing ProbEx record builder reduces remote ERA5; it does not save the full
historical archive locally. Its new 1979–2019 filenames do not replace the
1979–2021 records. Sources and exact archive names are listed in
`scripts/prepare_control_data.py`, from the
[WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
The four source stores' surface-variable metadata was checked when preparing
this update; no full weather archive was downloaded here.

### Pilot: retain the whole year, use 64 cells

Shortening the year changes the IDR fit. Instead the pilot uses geographically
spread land cells, while keeping all common dates for fitting and evaluation.

```bash
"$PCRPS_PYTHON" -u -m pcrps.compute_controls \
    --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
    --var t2m --leads 48 --pilot-cells 64 --output results/controls_pilot
"$PCRPS_PYTHON" scripts/plot_controls.py \
    --results results/controls_pilot --output figures/controls_pilot
```

The progress line reports completed cells, elapsed time and an approximate
remaining time. Compare pilot elapsed time per cell with the printed number of
eligible cells to estimate the full job. Fit complexity and disk access vary,
so this is a planning estimate, not a guaranteed completion time.

### Full experiment

```bash
mkdir -p logs
nohup bash jobs/run_controls.sh > logs/controls.log 2>&1 < /dev/null &
tail -f logs/controls.log
```

This runs both variables at 48/120/240 h, all four models, four representations
and depths 0/0.5/1/2. Heat and cold share each temperature fit. Four CPU processes
are used; do not launch many copies. Each completed variable/lead is saved
separately. If interrupted, rerun the same command; existing summaries are
replaced. Keep the full requested lead list so the common valid-time intersection
stays identical. A pilot with one lead may have additional valid dates.

### Optional held-out control on identical evaluation dates

Run BOTH modes on July–December, with the holdout fit confined to January–June.
This isolates reuse of verification observations more cleanly than comparing
the full-year in-sample score with a half-year holdout score. Seasonal changes
and smaller calibration size remain differences; report them.

```bash
"$PCRPS_PYTHON" -u -m pcrps.compute_controls \
    --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
    --var t2m --leads 48 120 240 --eval-start 2020-07-01 \
    --mode in-sample --output results/controls_jul_dec

"$PCRPS_PYTHON" -u -m pcrps.compute_controls \
    --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
    --var t2m --leads 48 120 240 --eval-start 2020-07-01 \
    --mode holdout --calibration-end 2020-06-30 --output results/controls_holdout
```

Repeat with `--var w10` for wind. Plot either output directory with the same
plotter. The NetCDF retains the exact calibration/verification timestamps,
selected cells and calibration maxima. No fair correction, bootstrap or
never-record baseline is introduced.

For an additional common-reference sensitivity, prepare HRES-fc0 with
`scripts/prepare_control_data.py --data-dir "$PCRPS_DATA_DIR" --include-hres-fc0`,
then run **every model** with `--truth HRES_fc0` into a separate output directory.
ERA5 records stay fixed. Check the stored verification dates before comparing
references; the two archives may have different coverage.

## Outputs and verification

`results/controls/` contains per-variable/lead NetCDF and CSV summaries. Plots go
to `figures/controls/`; `paired_differences.csv` gives like-for-like model and
within-model contrasts, including the record/nonrecord contributions. Negative
differences favour the first model/representation. Raw scores at deeper thresholds
usually decrease because events get rarer; that alone is not increasing skill.

Before delivery, exact scores were checked against the brute-force weighted
pairwise formula, native fair CRPS against its pairwise formula, contribution
identities and IDR's monotone-predictor invariance. Small synthetic input files
exercise the actual CLI and plotting paths; they are not weather results.
The full scientific comparisons still have to run on Biobot.

Only new control scripts, this guide, requirements and a README pointer are
added. The original package's scientific runners and saved-file conventions
remain available.
