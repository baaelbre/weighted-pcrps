#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

: "${PCRPS_DATA_DIR:?Set PCRPS_DATA_DIR to the compact 2020 inputs}"
: "${PCRPS_RECORDS_ROOT:?Set PCRPS_RECORDS_ROOT to the historical record/scale files}"
for variable in t2m w10; do
    "${PCRPS_PYTHON:-python}" -u -m pcrps.compute_controls \
        --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
        --models hres graphcast pangu fuxi --var "$variable" \
        --leads 48 120 240 --depths 0 0.5 1 2 --output results/controls
done
"${PCRPS_PYTHON:-python}" scripts/plot_controls.py --results results/controls
