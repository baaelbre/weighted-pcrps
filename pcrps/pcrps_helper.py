import numpy as np
import time
import xarray as xr
from isodisreg import idr

def get_points(predictions):
    return np.array(predictions.points)

def get_cdf(predictions):
    return np.array(predictions.ecdf)

def modify_points(points):
    return np.hstack([points[0], np.diff(points)])

def _pcrps(y, p, w, x):
    """
    Compute PCRPS for a single observation and a single IDR predictive distribution.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like
        Support points of the predictive distribution.

    Returns
    -------
    float
        PCRPS value for this observation and IDR predictive distribution.
    """
    return 2*np.sum(w*(np.array((y<=x))-p+0.5*w)*np.array(x-y))

def _qw_pcrps_upper(y, p, x, q):
    """
    Compute upper-tail quantile-weighted PCRPS (qwPCRPS) for a single observation and 
    a single IDR predictive distribution at given quantile levels.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like
        CDF values at support points x.
    x : array-like
        Support points of the predictive distribution.
    q : float or array-like
        One or more quantiles in (0, 1).

    Returns
    -------
    np.ndarray, shape (n_q,)
        qwPCRPS values for each quantile in `q`.
    """
    q = np.atleast_1d(np.asarray(q))
    p_star = np.maximum(p[None, :], q[:, None])
    
    p_prev = np.empty_like(p)
    p_prev[0] = 0.0
    p_prev[1:] = p[:-1]
    p_prev_star = np.maximum(p_prev[None, :], q[:, None])
    
    indicator = (y <= x).astype(float)[None, :]
    terms = indicator * (p_star - p_prev_star) - 0.5 * (p_star**2 - p_prev_star**2)
    
    return 2 * np.sum(terms * (x - y)[None, :], axis=1)

def _tw_pcrps_upper(y, p, w, x, t):
    """
    Compute upper-tail threshold-weighted PCRPS (twPCRPS) for a single observation and 
    a single IDR predictive distribution at given threshold levels.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like, shape (n_points,)
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like, shape (n_points,)
        Support points of the predictive distribution.
    t : array-like
        One or more thresholds.

    Returns
    -------
    np.ndarray, shape (n_q,)
        twPCRPS values for each threshold in `t`.
    """
    t = np.atleast_1d(t)
    x_clip = np.maximum(x[None, :], t[:, None])
    y_clip = np.maximum(y, t)

    w2 = w[None, :]
    p2 = p[None, :]
    indicator = (y_clip[:, None] <= x_clip).astype(float)
    terms = w2 * (indicator - p2 + 0.5 * w2) * (x_clip - y_clip[:, None])

    return 2 * np.sum(terms, axis=1)

def compute_easyuq(preds, obs, var, lead_time, level=None, compute_qw=False, quantiles=None, compute_tw=False, thresholds=None):    
    """
    Fit EasyUQ at each latitude/longitude point and compute PCRPS and optionally qwPCRPS and/or twPCRPS.

    Parameters
    ----------
    preds : xarray.DataArray
        Data array of model outputs for the variable of interest.
    obs : xarray.DataArray
        Observed data array for the variable of interest.
    var : str
        Name of the variable.
    lead_time : numpy.timedelta64
        Prediction lead time.
    level : int, optional
        Vertical level (if applicable) for the variable.
    compute_qw : bool, optional
        If True, also compute quantile-weighted PCRPS (qwPCRPS) at the specified quantile levels.
    quantiles : array-like, optional
        Sequence of quantile levels in [0, 1] at which qwPCRPS is evaluated.
    compute_tw : bool, optional
        If True, also compute threshold-weighted PCRPS (twPCRPS) at the specified thresholds.
    thresholds : array-like, optional
        Sequence of threshold levels at which twPCRPS is evaluated.

    Returns
    -------
    xarray.Dataset
        Dataset containing, for each latitude/longitude point (and optional level) and forecast 
        initialization time, PCRPS (and, if requested, qwPCRPS and twPCRPS) together with the corresponding 
        fitting time, PCRPS evaluation time, and, if computed, qwPCRPS and twPCRPS evaluation time.
    """
    # Initialize an empty list to store the datasets for each grid point
    pcrps_datasets = []
    # Iterate over each grid point in chunk
    for lat in preds.latitude.values:
        for lon in preds.longitude.values:
            # Select data
            preds_point = preds.sel(latitude=lat, longitude=lon, prediction_timedelta=lead_time)
            preds_point_df = preds_point.to_dataframe()[[var]]

            # Get valid times
            valid_time = preds_point.time + lead_time

            # Extract corresponding observations       
            y = obs.sel(latitude=lat, longitude=lon, time=valid_time).values

            start_time_idr = time.time()
            # Fit IDR
            fitted_idr = idr(y, preds_point_df)
            
            # Predict
            easyuq_preds_point = fitted_idr.predict(preds_point_df, digits=8)
            predictions = easyuq_preds_point.predictions
            
            if y.ndim > 1:
                raise ValueError('obs must be a 1-D array')
            if np.any(np.isnan(y)):
                raise ValueError('obs contains NaN values')
            if y.size != 1 and len(y) != len(predictions):
                raise ValueError('obs must have length 1 or the same length as predictions')

            x = list(map(get_points, predictions))
            p = list(map(get_cdf, predictions))
            w = list(map(modify_points, p))
            
            idr_time = np.float32(time.time() - start_time_idr)       
            
            # PCRPS
            start_time_pcrps = time.time()
            pcrps = np.float32((list(map(_pcrps, y, p, w, x))))
            pcrps_time = np.float32(time.time() - start_time_pcrps)

            da_pcrps = xr.DataArray(pcrps, dims=['time'], coords={'time': preds_point.time})

            # Expand dimensions
            expand_dims = {
                'longitude': [lon],
                'latitude': [lat],
                'prediction_timedelta': preds.prediction_timedelta
            }
            if level is not None:
                expand_dims['level'] = [level]
                
            da_pcrps = da_pcrps.expand_dims(expand_dims)
            
            vars_dict = {f'{var}_pcrps': da_pcrps}
            
            time_array = np.stack([idr_time, pcrps_time], axis=-1)
            time_dims = ['task']
            time_coords = {
                'task': ['idr_time', 'pcrps_time']
            }
            time_da = xr.DataArray(time_array, dims=time_dims, coords=time_coords)
            
            # qwPCRPS
            if compute_qw:
                q = np.asarray(quantiles, dtype=np.float32)

                start_time_qw_pcrps = time.time()
                qw_pcrps = np.float32(list(map(_qw_pcrps_upper, y, p, x, np.broadcast_to(q, (y.size, q.size)))))
                qw_pcrps_time = np.float32(time.time() - start_time_qw_pcrps)
                
                time_da = xr.concat([time_da, xr.DataArray([qw_pcrps_time], dims='task', coords={'task': ['qw_pcrps_time']})], dim='task')
                
                da_qw_pcrps = xr.DataArray(qw_pcrps, dims=['time', 'quantile'], coords={'time': preds_point.time, 'quantile': q})
                da_qw_pcrps = da_qw_pcrps.expand_dims(expand_dims)
                
                vars_dict[f'{var}_qw_pcrps'] = da_qw_pcrps
                
            # twPCRPS
            if compute_tw:
                t = thresholds.sel(latitude=lat, longitude=lon, time=valid_time).values

                start_time_tw_pcrps = time.time()
                tw_pcrps = np.float32(list(map(_tw_pcrps_upper, y, p, w, x, t)))
                tw_pcrps_time = np.float32(time.time() - start_time_tw_pcrps)
                
                time_da = xr.concat([time_da, xr.DataArray([tw_pcrps_time], dims='task', coords={'task': ['tw_pcrps_time']})], dim='task')
                
                da_tw_pcrps = xr.DataArray(tw_pcrps, dims=['time', 'threshold'], coords={'time': preds_point.time, 'threshold': thresholds.threshold})
                da_tw_pcrps = da_tw_pcrps.expand_dims(expand_dims)
                
                vars_dict[f'{var}_tw_pcrps'] = da_tw_pcrps

            time_da = time_da.expand_dims(expand_dims)    
            
            vars_dict[f'{var}_time'] = time_da

            pcrps_datasets.append(xr.Dataset(vars_dict))
            
    # Merge datasets
    pcrps_dataset = xr.merge(pcrps_datasets)
    
    return pcrps_dataset