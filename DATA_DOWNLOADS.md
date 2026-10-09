# Shared ProbEx data for the 2020 controls

The downloads have completed on Biobot. The EasyUQ evaluator reads them directly
from `/data/nvme1/bastiaan/probex-data`. There is no separate EasyUQ data folder.

Set `PCRPS_DATA_DIR` to that root, not to `forecasts/` or an individual store.
Set `PCRPS_RECORDS_ROOT` to its `era5_records/` directory.

| Dataset | Path relative to the data root |
| --- | --- |
| ERA5 2020 reference | `era5/ERA5_2020_0012_no_Antarctic_1p50.zarr` |
| HRES 2020 forecasts | `forecasts/HRES/hres_240x121_2020.zarr` |
| GraphCast 2020 forecasts | `forecasts/GraphCast/graphcast_240x121_2020.zarr` |
| Pangu 2020 forecasts | `forecasts/Pangu/pangu_240x121_2020.zarr` |
| FuXi 2020 forecasts | `forecasts/FuXi/fuxi_240x121_2020.zarr` |
| HRES-fc0 2020 reference | `forecasts/HRES_fc0/HRES_fc0_2020_0012_no_Antarctic_1p50.zarr` |
| Heat record + scale | `era5_records/ERA5_record_max_month_1979_2019_t2m.nc` |
| Cold record + scale | `era5_records/ERA5_record_min_month_1979_2019_t2m.nc` |
| Wind record + scale | `era5_records/ERA5_record_max_month_1979_2019_w10.nc` |
| Land-sea mask | `era5_records/ERA5_lsm_sfc.nc` |

The Zarr stores retain the long variable names `2m_temperature` and
`10m_wind_speed`. Each deterministic forecast store has both variables and leads
12/48/120/168/240 h. The control experiment uses 48/120/240 h. Historical NetCDFs
contain `record` and `scale`, with the latter the sample SD across 41 yearly
monthly extremes from 1979–2019. The 2022 experiment's 1979–2021 records are
not used for these 2020 controls.

`download_coverage_2020.json` describes the downloaded coverage and nonfinite
counts. FuXi has 702 initialization times in the source year, versus 732 for
other forecast stores. The evaluator intersects **valid dates across all
models and all requested leads**; it does not fill missing cases. It removes
exactly 60S from every input, then requires the remaining coordinates to match.
HRES and HRES-fc0 are distinct: a forecast system and a verification reference.

For a fresh machine, use the ProbEx repository's downloader and record builder:

```bash
cd /home/bastiaan/probex
export PROBEX_DATA=/data/nvme1/bastiaan/probex-data
/opt/miniconda3/envs/probex/bin/python -u download/download_control_data_wb2.py \
    --data-dir "$PROBEX_DATA" --models hres graphcast pangu fuxi \
    --leads 12 48 120 168 240 --include-hres-fc0 --jobs 6
bash bash_scripts/run_biobot_records_2020.sh
```

These commands are not needed again for the completed Biobot downloads. Monthly
download parts remain in `download_parts/2020/leads_12-48-120-168-240/` for resume.
The fork's older `scripts/prepare_control_data.py` remains available for its
legacy flat layout; use the ProbEx commands above for the shared layout.

Next: follow [CONTROLS.md](CONTROLS.md) for the environment, small pilot and full
experiment. No forecast generation or GPU is required.
