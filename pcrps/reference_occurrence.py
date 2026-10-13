"""Occurrence diagnostics for the matched raw 2020 forecasts.

These are classification/point-forecast diagnostics, not probabilistic scores.
All arrays below have already been aligned and checked by the existing reader.
"""
import numpy as np
import xarray as xr


def monthly_forecast_sd(forecasts, months):
    """SD across matched valid times in each calendar month (sample SD, ddof=1).

    This is the model's forecast variability, NOT the historical record scale s.
    forecasts: model x time x cell; months: zero-based valid-time months.
    """
    sd = np.full((len(forecasts), 12, forecasts.shape[-1]), np.nan)
    for month in np.unique(months):
        selected = months == month
        if selected.sum() < 2:
            raise ValueError("Need at least two matched times in each represented month for forecast SD.")
        sd[:, month] = forecasts[:, selected].std(axis=1, ddof=1)
        constant = np.ptp(forecasts[:, selected], axis=1) == 0
        sd[:, month][constant] = 0  # Avoid roundoff SD for a constant forecast.
    return sd


def occurrence_diagnostics(forecasts, observations, record, forecast_sd,
                           months, weights, models, references, sign, gains):
    """Counts, gain-based PR, binary correlations, and jointly forecast records."""
    ntime, ncell = record.shape
    total_count = ntime * ncell
    total_weight = float(ntime * weights.sum())
    observed = sign * (observations - record) > 0
    raw_forecast = sign * (forecasts - record) > 0
    observed_count = observed.sum(axis=(1, 2), dtype=np.int64)
    observed_weight = observed.sum(axis=1) @ weights
    shape = (len(references), len(models), len(gains), 4)
    counts = np.zeros(shape, dtype=np.int64)
    area = np.zeros(shape, dtype=float)

    # 1. Zhang's gain construction: shift each model by its local monthly SD.
    # Positive gain always moves toward the extreme: +g*SD for heat/wind,
    # -g*SD for cold. For cold this reverses the paper's gain sign; the symmetric
    # gain range gives the same set of shifted forecasts.
    for m, model in enumerate(models):
        distance = sign * (forecasts[m] - record)
        sd = forecast_sd[m, months]
        print(f"    PR: {model}, {len(gains)} gains", flush=True)
        for j, gain in enumerate(gains):
            predicted = distance + gain * sd > 0
            predicted_count = int(predicted.sum())
            predicted_weight = float(predicted.sum(axis=0) @ weights)
            for t in range(len(references)):
                hit = predicted & observed[t]
                tp = int(hit.sum())
                tp_weight = float(hit.sum(axis=0) @ weights)
                counts[t, m, j] = [tp, predicted_count-tp,
                                   observed_count[t]-tp,
                                   total_count-predicted_count-observed_count[t]+tp]
                area[t, m, j] = [tp_weight, predicted_weight-tp_weight,
                                 observed_weight[t]-tp_weight,
                                 total_weight-predicted_weight-observed_weight[t]+tp_weight]
    # Roundoff in weighted sums can produce tiny negative zeroes.
    if counts.min() < 0 or area.min() < -1e-8:
        raise ValueError("Invalid confusion table.")
    area = np.maximum(area, 0)
    out = xr.Dataset(coords=dict(truth=references, model=models, gain=gains,
                                 outcome=["TP", "FP", "FN", "TN"], weighting=["count", "area"]))
    dims = ["truth", "model", "gain", "outcome"]
    out["confusion_count"] = (dims, counts)
    out["confusion_weight"] = (dims, area)
    out["observed_count"] = ("truth", observed_count)
    out["observed_weight"] = ("truth", observed_weight)
    out["forecast_count"] = (["model", "gain"], counts[0, :, :, 0] + counts[0, :, :, 1])
    out["forecast_weight"] = (["model", "gain"], area[0, :, :, 0] + area[0, :, :, 1])

    # Do not replace undefined precision/recall/correlation with zero.
    tables = np.stack([counts.astype(float), area])
    tp, fp, fn, tn = [tables[..., j] for j in range(4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics = {
            "precision": tp / (tp+fp),
            "recall": tp / (tp+fn),
            "false_positive_rate": fp / (fp+tn),
            "f1": 2*tp / (2*tp+fp+fn),
            "phi": (tp*tn-fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
            "observed_rate": (tp+fn) / (tp+fp+fn+tn),
            "forecast_rate": (tp+fp) / (tp+fp+fn+tn),
        }
    # Constant binary indicators have undefined correlation even if weighted
    # sums differ from their total by floating-point roundoff.
    predicted_count = counts[..., 0] + counts[..., 1]
    target_count = counts[..., 0] + counts[..., 2]
    constant = ((predicted_count == 0) | (predicted_count == total_count)
                | (target_count == 0) | (target_count == total_count))
    metrics["phi"][:, constant] = np.nan
    for name, values in metrics.items():
        out[name] = (["weighting", "truth", "model", "gain"], values)

    # 2. Gain=0 indicator correlations between BOTH analyses and ALL forecasts.
    # These also retain the full ERA5/HRES-fc0 event agreement table.
    flags = np.concatenate([observed, raw_forecast])
    names = references + models
    source_count = flags.sum(axis=(1, 2), dtype=np.int64)
    source_weight = flags.sum(axis=1) @ weights
    joint_count = np.zeros((len(names), len(names)), dtype=np.int64)
    joint_weight = np.zeros_like(joint_count, dtype=float)
    for i in range(len(names)):
        for j in range(i, len(names)):
            both = flags[i] & flags[j]
            joint_count[i, j] = joint_count[j, i] = int(both.sum())
            joint_weight[i, j] = joint_weight[j, i] = float(both.sum(axis=0) @ weights)
    totals = np.array([total_count, total_weight])
    rates = np.stack([source_count, source_weight]) / totals[:, None]
    joint_rates = np.stack([joint_count, joint_weight]) / totals[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = rates * (1-rates)
        correlation = ((joint_rates - rates[:, :, None]*rates[:, None, :])
                       / np.sqrt(variance[:, :, None]*variance[:, None, :]))
    constant = (source_count == 0) | (source_count == total_count)
    correlation[:, constant, :] = np.nan
    correlation[:, :, constant] = np.nan
    out = out.assign_coords(source=names, other_source=names)
    out["source_count"] = ("source", source_count)
    out["source_weight"] = ("source", source_weight)
    out["joint_exceedance_count"] = (["source", "other_source"], joint_count)
    out["joint_exceedance_weight"] = (["source", "other_source"], joint_weight)
    out["indicator_correlation"] = (["weighting", "source", "other_source"], np.clip(correlation, -1, 1))

    # 3. Forecast-conditioned control: for each AI/HRES pair select the SAME
    # cases where BOTH raw forecasts exceed R. Observations do not select cases.
    # Each pair has its own subset; do not rank different AI models across pairs.
    if "hres" in models and len(models) > 1:
        comparisons = [m for m in models if m != "hres"]
        hres = models.index("hres")
        shape = (2, len(references), len(comparisons), 2)
        bias, mse = np.full(shape, np.nan), np.full(shape, np.nan)
        pair_count = np.zeros(len(comparisons), dtype=np.int64)
        pair_weight = np.zeros(len(comparisons))
        pair_times = np.zeros(len(comparisons), dtype=np.int64)
        for p, model in enumerate(comparisons):
            m = models.index(model)
            selected = raw_forecast[m] & raw_forecast[hres]
            pair_count[p] = selected.sum()
            pair_weight[p] = float(selected.sum(axis=0) @ weights)
            # This counts valid TIMES, not independent days or weather episodes.
            pair_times[p] = selected.any(axis=1).sum()
            if pair_count[p] == 0:
                continue
            selected_weights = np.broadcast_to(weights, record.shape)[selected]
            for t in range(len(references)):
                for role, index in enumerate([m, hres]):
                    error = forecasts[index][selected] - observations[t][selected]
                    bias[0, t, p, role] = error.mean()
                    mse[0, t, p, role] = np.mean(error**2)
                    if pair_weight[p] > 0:
                        bias[1, t, p, role] = np.sum(error*selected_weights)/pair_weight[p]
                        mse[1, t, p, role] = np.sum(error**2*selected_weights)/pair_weight[p]
        out = out.assign_coords(comparison_model=comparisons, forecast_role=["ai", "hres"])
        dims = ["weighting", "truth", "comparison_model", "forecast_role"]
        out["joint_forecast_bias"] = (dims, bias)
        out["joint_forecast_mse"] = (dims, mse)
        out["joint_forecast_rmse"] = (dims, np.sqrt(mse))
        out["joint_forecast_count"] = ("comparison_model", pair_count)
        out["joint_forecast_weight"] = ("comparison_model", pair_weight)
        out["joint_forecast_time_count"] = ("comparison_model", pair_times)
    out.attrs = dict(
        diagnostic="occurrence", occurrence_threshold="Historical ERA5 monthly record, depth k=0",
        gain_definition="forecast + sign*gain*monthly_forecast_sd; sign=-1 for cold, +1 otherwise",
        forecast_sd_definition="Sample SD across matched valid times, per model/grid cell/calendar month/lead",
        forecast_sd_ddof=1, forecast_sd_period="Matched evaluation forecasts; descriptive, not fitted operational calibration",
        forecast_sd_zero_policy="Zero SD leaves a constant forecast unchanged at every gain",
        weighting_definition="count: each grid-cell/time pair; area: cosine latitude on the same cases",
        confusion_definition="TP/FP/FN/TN of shifted forecast exceedance against the named reference exceedance",
        joint_forecast_definition="Both the named AI and HRES raw forecasts exceed R; same subset for both systems and references",
        interpretation="Raw deterministic diagnostics; finite gain sweep, no AUPRC or probability-calibration claim",
    )
    return out
