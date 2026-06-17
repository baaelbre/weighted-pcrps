"""
Script to compute ensemble CRPS or twCRPS results.
"""
from absl import app
from absl import flags
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from scores.probability import crps_for_ensemble, tail_tw_crps_for_ensemble

# Define command-line flags
PREDICTION_PATH = flags.DEFINE_string(
    'prediction_path',
    None,
    help='Path to forecasts to evaluate in Zarr format.',
)
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to ground-truth to evaluate in Zarr format.',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 timestamp (inclusive) at which to start.',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 timestamp (inclusive) at which to stop.',
)
VARIABLES = flags.DEFINE_list(
    'variables', 
    ['2m_temperature'], 
    help='Variables to compute ensemble score for.'
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path', 
    None, 
    help='Where to save results.'
)
CRPS_METHOD = flags.DEFINE_string(
    'crps_method',
    'ecdf',
    help='Either "ecdf" or "fair".'
)
TW_CRPS = flags.DEFINE_bool(
    'tw_crps',
    False,
    help='If True, compute threshold-weighted CRPS instead of standard CRPS.'
)
QUANTILE_THRESHOLDS_PATH = flags.DEFINE_string(
    'quantile_thresholds_path',
    None,
    help='Path to thresholds based on historical quantiles for twCRPS.'
)
MONTHLY_MAX_RECORDS_PATH = flags.DEFINE_string(
    'monthly_max_records_path',
    None,
    help='Path to thresholds based on historical monthly max records for twCRPS.'
)
MONTHLY_MIN_RECORDS_PATH = flags.DEFINE_string(
    'monthly_min_records_path',
    None,
    help='Path to thresholds based on historical monthly min records for twCRPS.'
)
QUANTILE_THRESHOLDS = flags.DEFINE_list(
    'quantile_thresholds',
    ['q000', 'q001', 'q005', 'q095', 'q099', 'q100'],
    help='Subset of quantile-based thresholds for twCRPS.'
)
TAIL = flags.DEFINE_string(
    'tail',
    'upper',
    help='Tail for twCRPS, e.g. "upper" or "lower".',
)
ENSEMBLE_MEMBER_DIM = flags.DEFINE_string(
    'ensemble_member_dim',
    'number',
    help='Dimension name that specifies the ensemble member.'
)
LEAD_TIMES_HOURS = flags.DEFINE_list(
    'lead_times_hours',
    [],
    help='Optional list of lead times in hours. If empty, use all available lead times.'
)

def open_zarr_auto(path, **kwargs):
    """
    Open a Zarr store from either a GCS or local path.

    For public WeatherBench2 stores on Google Cloud Storage, anonymous access is used. For local paths, no storage options are passed.

    Parameters
    ----------
    path : str
        Path to the Zarr store.
    **kwargs
        Additional keyword arguments passed to ``xarray.open_zarr``.

    Returns
    -------
    xarray.Dataset
        Opened Zarr dataset.
    """
    if str(path).startswith('gs://'):
        return xr.open_zarr(store=path, storage_options={'token': 'anon'}, **kwargs)
    return xr.open_zarr(store=path, **kwargs)

def build_thresholds(targets, variables):    
    """
    Construct the threshold dataset used for twCRPS evaluation.

    The threshold dataset is formed by concatenating any combination of monthly minimum thresholds, 
    selected quantile-based thresholds, and  monthly maximum thresholds along the ``threshold`` dimension.

    Parameters
    ----------
    targets : xarray.Dataset
        Target dataset.
    variables : list of str
        Variables for which thresholds are constructed.

    Returns
    -------
    xarray.Dataset
        Threshold dataset for twCRPS evaluation.

    Raises
    ------
    ValueError
        If no threshold source is provided.
    """
    if (
        QUANTILE_THRESHOLDS_PATH.value is None
        and MONTHLY_MIN_RECORDS_PATH.value is None
        and MONTHLY_MAX_RECORDS_PATH.value is None
    ):
        raise ValueError(
            'tw_crps=True but none of --quantile_thresholds_path, --monthly_min_records_path, or --monthly_max_records_path are set.'
        )

    threshold_pieces = []
    months = targets.time.dt.month

    if MONTHLY_MIN_RECORDS_PATH.value is not None:
        rec_min = open_zarr_auto(MONTHLY_MIN_RECORDS_PATH.value)[variables]
        rec_min_thr = (
            rec_min.sel(month=months)
            .expand_dims(threshold=['monthly_min'], axis=-1)
            .reset_coords('month', drop=True)
        )
        threshold_pieces.append(rec_min_thr)

    if QUANTILE_THRESHOLDS_PATH.value is not None:
        ts = open_zarr_auto(QUANTILE_THRESHOLDS_PATH.value)[variables]
        if QUANTILE_THRESHOLDS.value:
            ts = ts.sel(threshold=QUANTILE_THRESHOLDS.value)
        ts_thr = ts.expand_dims(time=targets.time)
        threshold_pieces.append(ts_thr)

    if MONTHLY_MAX_RECORDS_PATH.value is not None:
        rec_max = open_zarr_auto(MONTHLY_MAX_RECORDS_PATH.value)[variables]
        rec_max_thr = (
            rec_max.sel(month=months)
            .expand_dims(threshold=['monthly_max'], axis=-1)
            .reset_coords('month', drop=True)
        )
        threshold_pieces.append(rec_max_thr)

    thresholds = xr.concat(threshold_pieces, dim='threshold').chunk({
        'time': 1,
        'longitude': -1,
        'latitude': -1,
        'threshold': -1,
    })

    return thresholds

def main(argv):
    ens = open_zarr_auto(
        PREDICTION_PATH.value,
        decode_timedelta=True,
    ).sel(time=slice(TIME_START.value, TIME_STOP.value))[VARIABLES.value]
    ens = ens.drop_vars('lead_time_secs', errors='ignore')
    
    if LEAD_TIMES_HOURS.value:
        lead_times = [np.timedelta64(int(h), 'h') for h in LEAD_TIMES_HOURS.value]
        ens = ens.sel(prediction_timedelta=lead_times)
        
    ensemble_member_dim = ENSEMBLE_MEMBER_DIM.value
        
    ens = ens.chunk({
        'time': 1,
        'latitude': -1,
        'longitude': -1,
        'prediction_timedelta': 1,
        ensemble_member_dim: -1,
    })
    
    tmin = ens.time[0] + ens.prediction_timedelta.min()
    tmax = ens.time[-1] + ens.prediction_timedelta.max()
    
    targets = open_zarr_auto(
        TARGET_PATH.value,
        consolidated=True,
    ).sel(time=slice(tmin, tmax))[VARIABLES.value]
    
    targets = targets.chunk({
        'time': 1,
        'longitude': -1,
        'latitude': -1,
    })
    
    thresholds = None
    if TW_CRPS.value:
        thresholds = build_thresholds(targets, VARIABLES.value)
    
    ens_score_list = []
    
    for lt in ens.prediction_timedelta:
        # align times for this lead time
        valid_time = ens.time + lt

        lt_targets = targets.sel(time=valid_time)

        lt_ens_fct = ens.sel(prediction_timedelta=lt).assign_coords({'time': valid_time})
        
        if TW_CRPS.value:
            thresholds_lt = thresholds.sel(time=valid_time)

            ens_score_list.append(
                tail_tw_crps_for_ensemble(
                    lt_ens_fct,
                    lt_targets,
                    threshold=thresholds_lt,
                    tail=TAIL.value,
                    ensemble_member_dim=ensemble_member_dim,
                    method=CRPS_METHOD.value,
                    reduce_dims=['time'],
                )
            )
        else:
            ens_score_list.append(
                crps_for_ensemble(
                    lt_ens_fct,
                    lt_targets,
                    ensemble_member_dim=ensemble_member_dim,
                    method=CRPS_METHOD.value,
                    reduce_dims=['time'],
                )
            )
        
    ens_score = xr.concat(ens_score_list, dim='prediction_timedelta')
    
    write = ens_score.to_zarr(
        OUTPUT_PATH.value,
        mode='w',
        consolidated=True,
        compute=False,
    )

    with ProgressBar():
        dask.compute(write, num_workers=32)
    
if __name__ == '__main__':
    flags.mark_flags_as_required(['prediction_path', 'target_path', 'output_path'])
    app.run(main)