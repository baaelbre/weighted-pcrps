"""
Script to compute quantile-based and monthly record-based threshold datasets from ERA5.
"""
import logging
import os
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_wind_speed',
    'total_precipitation_24hr',
]

input_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
time_start = '1979-01-01'
time_stop = '2019-12-31'

def main():
    # expects PCRPS_THRESHOLDS_DIR to be set in the shell, e.g. via ``source local_paths.env``
    thresholds_dir = os.environ['PCRPS_THRESHOLDS_DIR']

    output_quantiles_path = f'{thresholds_dir}/era5_quantiles_1979_2019.zarr'
    output_quantiles_reduced_path = f'{thresholds_dir}/era5_quantiles_1979_2019_reduced.zarr'
    output_monthly_max_path = f'{thresholds_dir}/era5_record_max_month_1979_2019.zarr'
    output_monthly_min_path = f'{thresholds_dir}/era5_record_min_month_1979_2019.zarr'    
    
    logging.info('Opening ERA5 dataset.')
    era5 = xr.open_zarr(
        store=input_path,
        storage_options={'token': 'anon'},
        decode_timedelta=True,
    ).sel(time=slice(time_start, time_stop))[variables]
    
    era5 = era5.chunk({'time': -1, 'latitude': 11, 'longitude': 60})

    logging.info('Building quantile-based thresholds.')
    qs = np.arange(0, 101, dtype=np.float64) / 100.0
    q_labels = [f'q{i:03d}' for i in range(101)]

    era5_quantiles = (
        era5
        .quantile(q=qs, dim='time', keep_attrs=True)
        .assign_coords(quantile=q_labels)
        .rename(quantile='threshold')
        .astype('float32')
    )

    era5_quantiles_reduced = era5_quantiles.sel(
        threshold=['q000', 'q001', 'q099', 'q100']
    )
    
    logging.info('Building monthly record thresholds.')
    era5_record_max_month = era5.groupby('time.month').max(dim='time', skipna=True).astype('float32')
    era5_record_min_month = era5.groupby('time.month').min(dim='time', skipna=True).astype('float32')
    
    logging.info('Creating lazy zarr write tasks.')
    write_q = era5_quantiles.to_zarr(
        output_quantiles_path,
        mode='w',
        consolidated=True,
        compute=False,
    )
    write_q_reduced = era5_quantiles_reduced.to_zarr(
        output_quantiles_reduced_path,
        mode='w',
        consolidated=True,
        compute=False,
    )
    write_max = era5_record_max_month.to_zarr(
        output_monthly_max_path,
        mode='w',
        consolidated=True,
        compute=False,
    )
    write_min = era5_record_min_month.to_zarr(
        output_monthly_min_path,
        mode='w',
        consolidated=True,
        compute=False,
    )
    
    logging.info('Executing all writes in one graph.')
    with ProgressBar():
        dask.compute(write_q, write_q_reduced, write_max, write_min)

    logging.info('Finished.')
    
if __name__ == '__main__':
    main()