"""Exact scores for EasyUQ distributions and their point-forecast controls.

Support points have unequal probability weights. They are NOT exchangeable
ensemble members: neither Monte Carlo sampling nor a fair correction is used.
"""
import numpy as np
import pandas as pd
from isodisreg import idr
from pcrps.pcrps_helper import _pcrps

REPRESENTATIONS = ["raw", "easyuq_mean", "easyuq_median", "easyuq"]
SCORES = ["crps", "rtwcrps", "brier", "bias", "squared_error", "probability"]


def distribution_scores(x, weights, observation, thresholds, sign):
    """Return score x depth for one weighted discrete forecast, upper or lower tail."""
    x, weights = np.asarray(x, dtype=float), np.asarray(weights, dtype=float)
    if np.any(weights < -1e-10) or not np.isclose(weights.sum(), 1, atol=1e-7):
        raise ValueError("Invalid EasyUQ probability weights.")
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    p = weights.cumsum()
    crps = _pcrps(observation, p, weights, x)
    # Both max and min are monotone, so the original support ordering remains valid.
    transformed = (np.maximum if sign == 1 else np.minimum)(x[None, :], thresholds[:, None])
    obs = (np.maximum if sign == 1 else np.minimum)(observation, thresholds)
    tw = 2 * np.sum(weights * ((obs[:, None] <= transformed) - p + 0.5 * weights) * (transformed - obs[:, None]), axis=1)
    probability = ((sign * (x[None, :] - thresholds[:, None]) > 0) * weights).sum(axis=1)
    event = sign * (observation - thresholds) > 0
    error = np.dot(weights, x) - observation
    return np.stack([np.full(len(thresholds), crps), tw, (probability - event) ** 2,
                     np.full(len(thresholds), error), np.full(len(thresholds), error ** 2), probability])


def score_cell(payload):
    """Fit once per model/cell/lead; score raw, recalibrated centres and full CDFs.

    Return time sums, observed-event sums and counts; global reductions then
    use exactly the same cosine weights and denominators for every model.
    """
    forecasts, truth, months, records, scales, depths, signs, fit, evaluate = payload
    n_models, _ = forecasts.shape
    n_events, n_depths = len(signs), len(depths)
    totals = np.zeros((n_models, n_events, 4, len(SCORES), n_depths))
    events = np.zeros_like(totals)
    counts = np.zeros((n_events, n_depths), dtype=int)
    fit_max = np.full(n_models, truth[fit].max())
    for j, sign in enumerate(signs):
        threshold = records[j, months[evaluate]] + sign * np.asarray(depths)[:, None] * scales[j, months[evaluate]]
        counts[j] = (sign * (truth[evaluate] - threshold) > 0).sum(axis=1)
    for model in range(n_models):
        predictor = pd.DataFrame({"forecast": forecasts[model]})
        fitted = idr(truth[fit], predictor.loc[fit].reset_index(drop=True))
        predictions = fitted.predict(predictor.loc[evaluate].reset_index(drop=True), digits=8).predictions
        for index, prediction in zip(np.flatnonzero(evaluate), predictions):
            x = np.asarray(prediction.points, dtype=float)
            cdf = np.asarray(prediction.ecdf, dtype=float)
            w = np.diff(np.r_[0, cdf])
            mean = np.dot(x, w)
            median = x[np.searchsorted(cdf, 0.5, side="left")]
            distributions = [(np.array([forecasts[model, index]]), np.ones(1)),
                             (np.array([mean]), np.ones(1)), (np.array([median]), np.ones(1)), (x, w)]
            for j, sign in enumerate(signs):
                threshold = records[j, months[index]] + sign * np.asarray(depths) * scales[j, months[index]]
                event = sign * (truth[index] - threshold) > 0
                for rep, (points, weights) in enumerate(distributions):
                    scores = distribution_scores(points, weights, truth[index], threshold, sign)
                    totals[model, j, rep] += scores
                    events[model, j, rep] += scores * event
    return totals, events, counts, fit_max
