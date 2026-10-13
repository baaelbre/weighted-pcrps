# Remaining 2020 record diagnostics

This extends the existing [reference comparison](REFERENCE_COMPARISON.md) with
the occurrence and forecast-conditioned diagnostics discussed for Zhang et al.
It uses the same local HRES, GraphCast, Pangu and FuXi forecasts, both ERA5 and
HRES-fc0 references, and fixed ERA5 1979–2019 monthly thresholds.

## Run on Biobot

After copying this update into your repository, committing/pushing it, and pulling
it on Biobot, run:

```bash
cd /home/bastiaan/weighted-pcrps
git pull --ff-only
export PCRPS_PYTHON=/opt/miniconda3/envs/probex-easyuq/bin/python
export PCRPS_DATA_DIR=/data/nvme1/bastiaan/probex-data
export PCRPS_RECORDS_ROOT="$PCRPS_DATA_DIR/era5_records"
mkdir -p logs
PCRPS_OCCURRENCE_LOG="logs/reference_occurrence_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash jobs/run_reference_occurrence.sh \
    > "$PCRPS_OCCURRENCE_LOG" 2>&1 < /dev/null &
echo "Started PID: $!"
echo "Log: $PCRPS_OCCURRENCE_LOG"
tail -f "$PCRPS_OCCURRENCE_LOG"
```

Ctrl+C stops the log view, not the background run. No downloads, GPU, EasyUQ
refitting or new dependencies are needed. Inputs stay under `/data/nvme1`.
Results go to `results/reference_occurrence/`; plots and CSV tables go to
`figures/reference_occurrence/`. These paths are relative to this repository.
The earlier reference comparison and EasyUQ outputs are untouched.

The run reads each variable/lead once, with four Dask read threads. Variables and
leads run sequentially. Dask bars describe reads; scoring prints progress for
each event/model, followed by elapsed time per lead. Use the first completed
lead's time to estimate the rest; no IDR fitting is done.

## What is added

Defaults: temperature heat/cold and wind; 48, 120 and 240 h; land; 2020 at 00/12
UTC. The reader intersects valid dates across both references, all four models
and all requested leads. It retains the same grid and positive historical-scale
mask as the previous margin/depth run. No nearest-grid alignment or model-specific
dropping of missing values is allowed.

1. **Raw record counts and confusion matrices.** Compare every forecast to the
   same historical record and verify its binary indicator against each reference.
   Save TP, FP, FN, TN, precision, recall, F1 and phi (binary/Matthews correlation).
   Save ordinary counts and cosine-area-weighted versions. Counts refer to
   grid-cell/time pairs, not monthly blocks or independent weather events.
2. **Precision–recall curves.** Apply the paper's gain construction, with 61 gains
   from -1.5 to +1.5. Shift each forecast by its own local monthly forecast SD.
   Positive gain moves toward the extreme: add gain*SD for heat/wind, subtract it
   for cold. The cold gain label is reversed relative to the paper's additive
   formula, but the symmetric range produces the same set of shifts.
3. **Indicator correlations.** A full matrix includes the two references and
   all four raw forecasts. The reference-agreement table also saves how many
   cases are flagged by both references, either alone, or neither.
4. **Jointly forecast records.** For each AI/HRES pair, select cases where BOTH
   raw forecasts exceed the historical record. Score both systems on that same
   subset against each reference. Save bias, MSE, RMSE, case counts and the
   number of valid times containing at least one selected cell. Each AI/HRES
   pair has its own subset: do not compare different AI models across pairs.

For classification there is one binary target per reference, not four E/T
combinations: no separate continuous T is needed to count hits or false alarms.
The previous conditional-error experiment still separates E and T. The joint
forecast control selects on forecasts alone, then independently changes T.

## Important definitions

The **forecast SD for the PR sweep is not the historical scale s used for depth**.
It is the sample SD (`ddof=1`) over matched valid times in a calendar month,
computed separately for every model, location and lead. The paper specifies a
monthly forecast SD but not its degrees-of-freedom convention; ours is recorded
explicitly. This is a descriptive evaluation-period adjustment, not an operational
postprocessor trained on independent data. A constant forecast remains unchanged
for all gains. Both occurrence and joint-forecast selection use strict record
exceedance (value < record for cold), at depth k=0.

The PR sweep is limited to the stated gain range: it is not guaranteed to span
recall 0 to 1. We do not report an AUPRC/average-precision value for this truncated
curve. The figures show both count and area weighting, a dot at gain=0, and a
horizontal prevalence line. Undefined precision, recall and correlations are
NaN. Row percentages in confusion plots include the raw cell counts.

This remains a **harmonized 1.5-degree 2020 sensitivity study**, not an exact
reproduction of the paper's 0.25-degree experiment, record period or operational
model variants. No extra inference, uncertainty quantification or independence
tests are included. Neither diagnostic identifies a structural soft cap by itself.

## Files and figures

The evaluator writes nine NetCDF files (three event types x three leads). Each
retains case/record fingerprints, exact dates/cells, source paths, both weighting
conventions and the monthly forecast SDs needed to reconstruct the gain sweep.

The sequential Matplotlib script produces:

- PR curves, confusion matrices and indicator-correlation matrices per event/lead;
- predicted/observed record counts and weighted occurrence rates versus lead;
- jointly forecast record errors versus lead for each AI/HRES pair;
- six tables: `precision_recall.csv`, `occurrence_summary.csv` (gain=0),
  `record_counts.csv`, `indicator_correlations.csv`, `reference_agreement.csv`,
  and `joint_forecast_scores.csv`.

If matching old depth files exist in `results/reference_comparison/`, the plotter
checks their case/record fingerprints and k=0 event counts against the new run.
Different runs should have different folders. A pilot should always use separate
result and figure folders; never mix pilot and full results.

To remake the figures without reading forecasts again:

```bash
"$PCRPS_PYTHON" scripts/plot_reference_occurrence.py
```

For a quick execution check on 32 spread-out cells, retaining all common dates:

```bash
"$PCRPS_PYTHON" -u -m pcrps.compute_reference_comparison \
    --var t2m --leads 48 120 240 --diagnostics occurrence --pilot-cells 32 \
    --output results/occurrence_pilot
"$PCRPS_PYTHON" scripts/plot_reference_occurrence.py \
    --results results/occurrence_pilot --output figures/occurrence_pilot \
    --conditional-results results/occurrence_pilot
```

The existing evaluator defaults to the original conditional diagnostics. Use
`--diagnostics conditional occurrence` to generate both sets in one read pass
on a fresh run. Both plot scripts can read such a directory using `--results`.

Method reference: Zhang et al. (2026), *Physics-based models outperform AI weather
forecasts of record-breaking extremes*, Science Advances 12, eaec1433,
doi:10.1126/sciadv.aec1433; Methods Eq. (5) and supplementary Fig. S9.
