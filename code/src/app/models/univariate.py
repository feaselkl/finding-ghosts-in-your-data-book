# Finding Ghosts in Your Data
# Univariate statistical anomaly detection
# For more information on this, review chapters 6-9

import pandas as pd
import numpy as np
from pandas.core import base
from statsmodels import robust

def detect_univariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    # Standard deviation is not a very robust measure, so we weigh this lowest.
    # IQR is a reasonably good measure, so we give it the second-highest weight.
    # MAD is a robust measure for deviation, so we give it the highest weight.
    # The normal distribution tests are generally pretty good if we have the right
    # shape of the data and the correct number of observations.
    # The reason Grubbs' and Dixon's tests are so low is that they capture at most
    # 1 (Grubbs) or 2 (Dixon) outliers.
    weights = {"sds": 0.25, "iqrs": 0.35, "mads": 0.45,
               "grubbs": 0.05, "dixon": 0.15, "gesd": 0.3,
               "gaussian_mixture": 1.5}

    if (df['value'].count() < 3):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a minimum of at least three data points for anomaly detection.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, tests_run, diagnostics) = run_tests(df)
        df_scored = score_results(df_tested, tests_run, weights)
        df_out = determine_outliers(df_scored, sensitivity_score, max_fraction_anomalies)
        return (df_out, weights, { "message": "Ensemble of univariate statistical tests.", "Test diagnostics": diagnostics})

def run_tests(df):
    # Get our baseline calculations, prior to any data transformations.
    base_calculations = perform_statistical_calculations(df['value'])

    diagnostics = { "Base calculations": base_calculations }

    # for each test, execute and add a new score
    # Initial tests should NOT use the fitted calculations.
    b = base_calculations
    df['sds'] = [check_sd(val, b["mean"], b["sd"], 3.0) for val in df['value']]
    df['mads'] = [check_mad(val, b["median"], b["mad"], 3.0) for val in df['value']]
    df['iqrs'] = [check_iqr(val, b["median"], b["p25"], b["p75"], b["iqr"], 1.5) for val in df['value']]
    tests_run = {
        "sds": 1,
        "mads": 1,
        "iqrs": 1
    }
    
    diagnostics["Tests Run"] = tests_run

    return (df, tests_run, diagnostics)

def perform_statistical_calculations(col):
    mean = col.mean()
    sd = col.std()
    # Inter-Quartile Range (IQR) = 75th percentile - 25th percentile
    p25 = np.quantile(col, 0.25)
    p75 = np.quantile(col, 0.75)
    iqr = p75 - p25
    median = col.median()
    # Median Absolute Deviation (MAD)
    mad = robust.mad(col)
    min = col.min()
    max = col.max()
    len = col.shape[0]

    return { "mean": mean, "sd": sd, "min": min, "max": max,
        "p25": p25, "median": median, "p75": p75, "iqr": iqr, "mad": mad, "len": len }

def check_sd(val, mean, sd, min_num_sd):
    return check_stat(val, mean, sd, min_num_sd)

def check_mad(val, median, mad, min_num_mad):
    return check_stat(val, median, mad, min_num_mad)

def check_stat(val, midpoint, distance, n):
    # In the event that we are less than n times the distance
    # beyond the midpoint (mean or median) return the 
    # percent of the way we are to the extremity measure
    # and let the weighting process
    # figure out what to make of it.
    # If distance is 0, then distance-based calculations aren't meaningful--there is no spread.
    if (abs(val - midpoint) < (n * distance)):
        return abs(val - midpoint)/(n * distance)
    else:
        return 1.0

def check_iqr(val, median, p25, p75, iqr, min_iqr_diff):
    # We only want to check one direction, based on whether
    # the value is below the median or at/above.
    if (val < median):
        # If the value is in the p25-median range, it's
        # definitely not an outlier.  Return a value of 0.
        if (val > p25):
            return 0.0
        # If the value is between p25 and the outlier break point,
        # return a fractional score representing how distant it is.
        elif (p25 - val) < (min_iqr_diff * iqr):
            return abs(p25 - val)/(min_iqr_diff * iqr)
        # If the value is far enough away that it's definitely
        # an outlier, return 1.0
        else:
            return 1.0
    else:
        # If the value is in the median-p75 range, it's
        # definitely not an outlier.  Return a value of 0.
        if (val < p75):
            return 0.0
        # If the value is between p75 and the outlier break point,
        # return a fractional score representing how distant it is.
        elif (val - p75) < (min_iqr_diff * iqr):
            return abs(val - p75)/(min_iqr_diff * iqr)
        # If the value is far enough away that it's definitely
        # an outlier, return 1.0
        else:
            return 1.0

def score_results(df, tests_run, weights):
    # Because some tests do not run, we want to factor that into our anomaly score.
    tested_weights = {w: weights.get(w, 0) * tests_run.get(w, 0) for w in set(weights).union(tests_run)}

    # If a test was not run, its tests_run[] result will be 0, so we won't include it in our score.
    # If we divide by max weight, we end up with possible values of [0-1].
    # Multiplying by 0.95 makes it a little more likely that we mark an item as an outlier.
    return df.assign(anomaly_score=(
       df['sds'] * tested_weights['sds'] +
       df['iqrs'] * tested_weights['iqrs'] +
       df['mads'] * tested_weights['mads']
    ))

def determine_outliers(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    # Convert sensitivity score to be approximately the same
    # scale as anomaly score.  Note that sensitivity score is "reversed",
    # such that 100 is the *most* sensitive.
    sensitivity_score = (100 - sensitivity_score) / 100.0
    # Get the 100-Nth percentile of anomaly score.
    # Ex:  if max_fraction_anomalies = 0.1, get the
    # 90th percentile anomaly score.
    max_fraction_anomaly_score = np.quantile(df['anomaly_score'], 1.0 - max_fraction_anomalies)
    # If the max fraction anomaly score is greater than
    # the sensitivity score, it means that we have MORE outliers
    # than our max_fraction_anomalies supports, and therefore we
    # need to cut it off before we get down to our sensitivity score.
    # Otherwise, sensitivity score stays the same and we operate as normal.
    if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
        sensitivity_score = max_fraction_anomaly_score
    return df.assign(is_anomaly=(df['anomaly_score'] >= sensitivity_score))
