# Parallel 2020 data preparation

The updated `scripts/prepare_control_data.py` writes the same six compact stores
as before. The scoring and EasyUQ code do not change. Data preparation works in
the existing `probex` environment, without installing IDR or using a GPU.

```bash
cd /home/bastiaan/weighted-pcrps
mkdir -p logs
nohup /opt/miniconda3/envs/probex/bin/python -u scripts/prepare_control_data.py \
    --data-dir /data/nvme1/bastiaan/probex-data/easyuq_2020 \
    --models hres graphcast pangu fuxi \
    --leads 12 48 120 168 240 --include-hres-fc0 --jobs 6 \
    > logs/download_2020.log 2>&1 < /dev/null &
tail -f logs/download_2020.log
```

This reads T2m and precomputed W10, 00/12 UTC, on the supplied 1.5-degree grid
north of 60S. ERA5 and HRES-fc0 are verification fields; HRES is also downloaded
separately as a deterministic forecast. All four forecast systems use the same
requested leads. Records/scales for evaluation in 2020 must end in 2019.

| Output | Purpose |
| --- | --- |
| `era5_240x121_eval_times.zarr` | ERA5 2020 reference |
| `ifs_analysis_240x121_eval_times.zarr` | HRES-fc0 2020 reference |
| `hres_240x121_2020.zarr` | HRES forecasts |
| `graphcast_240x121_2020.zarr` | GraphCast forecasts |
| `pangu_240x121_2020.zarr` | Pangu forecasts |
| `fuxi_240x121_2020.zarr` | FuXi forecasts |

There are 144 jobs (six datasets × two variables × twelve months), with six
running concurrently. Each uses two Dask threads. All requested leads are kept
in the same job because FuXi stores all leads in each source chunk. Source data
are accessed anonymously over HTTPS with TLS verification enabled.

Each job writes a separate temporary store, checks its values and records a
checksum. Failed jobs retry up to three times. The same command reuses verified
monthly pieces and final files; it does not need to restart a full year.
One final local merge per dataset preserves the evaluator's existing filenames.
No tasks write concurrently into the same store. A lock prevents two launchers
from using the same data directory.

`download_coverage.json` reports missing dates and nonfinite values. Source gaps
are retained/reported, not filled with invented data. In the live metadata check,
FuXi has 702 initializations in 2020, versus 732 for the other archives. The
evaluator must still intersect valid dates and reject incomplete fields.
Archive coordinate roundoff can affect inclusion of exactly 60S; retain the
same record-builder convention and inspect the saved coverage before scoring.

Final files occupy approximately 3.1 GB uncompressed. Allow twice that while
monthly parts coexist; network transfer can be larger because unwanted leads
share source chunks. Delete `download_parts/` only after checking the final
stores. Existing outputs from the old downloader, which lack checksum markers,
are rebuilt from verified parts the first time, then reused normally.

`--plan` prints source metadata and job counts without downloading weather
fields. Progress counts completed jobs, not bytes or an exact ETA.

For the combined **Globus uCast/AWG + 2020 WeatherBench + 1979–2019 records/scales**
workflow, use the updated ProbEx package's `DATA_DOWNLOADS.md` and run
`bash bash_scripts/run_biobot_data_downloads.sh` from the ProbEx root.
This also documents the one-time interactive Globus destination setup.

Source URLs: [WeatherBench2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
