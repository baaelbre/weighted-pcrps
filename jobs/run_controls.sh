#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

: "${PCRPS_DATA_DIR:?Set PCRPS_DATA_DIR to the shared ProbEx data root}"
: "${PCRPS_RECORDS_ROOT:?Set PCRPS_RECORDS_ROOT to the historical record/scale files}"
for variable in t2m w10; do
    "${PCRPS_PYTHON:-python}" -u -m pcrps.compute_controls \
        --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
        --models hres graphcast pangu fuxi --var "$variable" \
        --year 2020 --record-tag 1979_2019 --truth ERA5 --surface land \
        --mode in-sample --leads 48 120 240 --depths 0 0.5 1 2 \
        --output results/controls
done
"${PCRPS_PYTHON:-python}" scripts/plot_controls.py --results results/controls
