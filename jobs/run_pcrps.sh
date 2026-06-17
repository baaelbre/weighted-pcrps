#!/usr/bin/env bash
set -eu

# Change to the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load machine-specific directory paths from local_paths.env.
if [ -f "${REPO_ROOT}/local_paths.env" ]; then
  set -a
  source "${REPO_ROOT}/local_paths.env"
  set +a
else
  echo "Missing local_paths.env in ${REPO_ROOT}"
  exit 1
fi

# -------------------------
# Fixed settings
# -------------------------
TIME_START="2020-01-01"
TIME_STOP="2020-12-16"

CHUNK_LON=8
CHUNK_LAT=11

LEAD_TIMES_HOURS="24,72,120,168,240"

QW_PCRPS="True"
TW_PCRPS="True"

QUANTILE_THRESHOLDS="${PCRPS_THRESHOLDS_DIR}/era5_quantiles_1979_2019.zarr"
MAXREC="${PCRPS_THRESHOLDS_DIR}/era5_record_max_month_1979_2019.zarr"
MINREC="${PCRPS_THRESHOLDS_DIR}/era5_record_min_month_1979_2019.zarr"

TEMPORAL_MEAN="True"

ERA5_TARGET="${PCRPS_DATA_DIR}/era5_240x121_eval_times.zarr"
IFS_TARGET="${PCRPS_DATA_DIR}/ifs_analysis_240x121_eval_times.zarr"

DIRECT_WORKERS=32
DIRECT_MODE="multi_processing"

TOTAL_JOBS=14
JOB=0

# /dev/null for quiet console
BEAM_OUT="/dev/null"

# -------------------------
# Run one job + print minimal status + timing
# -------------------------
run_job () {
  IDX=$1
  NAME=$2
  PRED=$3
  TARGET=$4
  OUT=$5
  VARS=$6
  
  # Skip if output Zarr already exists
  if [ -d "${OUT}" ]; then
    echo ""
    echo "============================================================"
    echo "SKIP  Job ${IDX}/${TOTAL_JOBS}: ${NAME} (output exists)"
    echo "Output:   ${OUT}"
    echo "============================================================"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "START Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Started:  $(date '+%F %T')"
  echo "Output:   ${OUT}"
  echo "============================================================"

  START=$(date +%s)

  # Run the command, but suppress its output:
  python -m pcrps.compute_pcrps \
    --prediction_path="${PRED}" \
    --target_path="${TARGET}" \
    --output_path="${OUT}" \
    --variables="${VARS}" \
    --time_start="${TIME_START}" \
    --time_stop="${TIME_STOP}" \
    --chunk_size_lon="${CHUNK_LON}" \
    --chunk_size_lat="${CHUNK_LAT}" \
    --qw_pcrps="${QW_PCRPS}" \
    --tw_pcrps="${TW_PCRPS}" \
    --quantile_thresholds_path="${QUANTILE_THRESHOLDS}" \
    --monthly_max_records_path="${MAXREC}" \
    --monthly_min_records_path="${MINREC}" \
    --temporal_mean="${TEMPORAL_MEAN}" \
    --lead_times_hours="${LEAD_TIMES_HOURS}" \
    --runner=DirectRunner \
    -- \
    --direct_num_workers="${DIRECT_WORKERS}" \
    --direct_running_mode="${DIRECT_MODE}" \
    > "${BEAM_OUT}" 2>&1

  END=$(date +%s)
  DUR=$((END - START))

  echo "DONE  Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Finished: $(date '+%F %T')"
  echo "Duration: ${DUR} seconds"
  echo "============================================================"
}

# -------------------------
# Jobs 
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "fuxi vs era5" \
  "${PCRPS_DATA_DIR}/fuxi_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/fuxi_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "fuxi vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/fuxi_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/fuxi_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast vs era5" \
  "${PCRPS_DATA_DIR}/graphcast_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/graphcast_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/graphcast_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/graphcast_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast_operational vs era5" \
  "${PCRPS_DATA_DIR}/graphcast_operational_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/graphcast_operational_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast_operational vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/graphcast_operational_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/graphcast_operational_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "pangu vs era5" \
  "${PCRPS_DATA_DIR}/pangu_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pangu_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "pangu vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/pangu_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pangu_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "pangu_operational vs era5" \
  "${PCRPS_DATA_DIR}/pangu_operational_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pangu_operational_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "pangu_operational vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/pangu_operational_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pangu_operational_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "hres vs era5" \
  "${PCRPS_DATA_DIR}/hres_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/hres_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "hres vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/hres_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/hres_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

JOB=$((JOB+1))
run_job "${JOB}" "era5_climatology vs era5" \
  "${PCRPS_DATA_DIR}/era5_climatology_forecasts_240x121_2020.zarr" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/era5_climatology_240x121_vs_era5.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "era5_climatology vs ifs_analysis" \
  "${PCRPS_DATA_DIR}/era5_climatology_forecasts_240x121_2020.zarr" \
  "${IFS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/era5_climatology_240x121_vs_ifs_analysis.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"
