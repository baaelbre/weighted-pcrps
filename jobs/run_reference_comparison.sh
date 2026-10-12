#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PCRPS_DATA_DIR="${PCRPS_DATA_DIR:-/data/nvme1/bastiaan/probex-data}"
export PCRPS_RECORDS_ROOT="${PCRPS_RECORDS_ROOT:-$PCRPS_DATA_DIR/era5_records}"

for variable in t2m w10; do
    "${PCRPS_PYTHON:-python}" -u -m pcrps.compute_reference_comparison \
        --data-dir "$PCRPS_DATA_DIR" --records-root "$PCRPS_RECORDS_ROOT" \
        --models hres graphcast pangu fuxi --var "$variable" \
        --year 2020 --record-tag 1979_2019 --record-hours 0 12 \
        --leads 48 120 240 --surface land --axes margin depth \
        --margins 0 0.5 1 2 3 4 5 6 --depths 0 0.5 1 2 \
        --output results/reference_comparison
done
"${PCRPS_PYTHON:-python}" scripts/plot_reference_comparison.py \
    --results results/reference_comparison --output figures/reference_comparison
