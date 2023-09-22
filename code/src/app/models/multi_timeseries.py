# Finding Ghosts in Your Data
# Multiple time series anomaly detection
# For more information on this, review chapters 16-17

import pandas as pd
import numpy as np
from pandas.core import base

def detect_multi_timeseries(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    weights = { "DIFFSTD": 1.0 }

    # Ensure that everything is sorted by dt and series key
    df = df.sort_values(["dt", "series_key"], axis=0, ascending=True)
    
    num_series = len(df["series_key"].unique())
    num_data_points = df['value'].count()
    if (num_data_points / num_series < 15):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least fifteen data points per time series for anomaly detection.  You sent {num_data_points} per series.")
    elif (num_series < 2):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least two time series for anomaly detection.  You sent {num_series} series.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, tests_run, diagnostics) = run_tests(df)
        (df_scored, diag_scored) = score_results(df_tested, tests_run, sensitivity_score)
        (df_out, diag_outliers) = determine_outliers(df_scored, max_fraction_anomalies)
        return (df_out, weights, { "message": "Result of single time series statistical tests.", "Tests run": tests_run, "Test diagnostics": diagnostics, "Outlier scoring": diag_scored, "Outlier determination": diag_outliers})

def run_tests(df):
    tests_run = {
        "DIFFSTD": 1
    }

    # Break out each series to operate on individually.
    series = [y for x, y in df.groupby("series_key", as_index=False)]
    # Grab basic information:  number of series, length of series.
    num_series = len(series)
    l = len(series[0])

    diagnostics = {
        "Number of time series": num_series,
        "Time series length": l
    }

    # Break out the series into segments of approximately 7 data points.
    # 7 data points allows us to have at least 2 segments given our 15-point minimum.
    # We use integer math here to ensure no segment has just 1-2 records and no segment
    # is wildly unbalanced in size compared to the others.  At a minimum,
    # we should have 6 data points per segment.  At a maximum, we can end up with 10.
    series_segments = [np.array_split(series[x], (l // 7)) for x in range(num_series)]
    num_segments = len(series_segments[0])
    num_records = df['key'].shape[0]

    diagnostics["Number of records"] = num_records
    diagnostics["Number of segments per time series"] = num_segments

    segment_means = generate_segment_means(series_segments, num_series, num_segments)
    diagnostics["Segment means"] = segment_means
    segments_diffstd = check_diffstd(series_segments, segment_means, num_series, num_segments)
    # Merge segments together as df.  Segments comes in as a list of lists, each of which contains a DataFrame.
    # First, flatten out the list of lists, giving us a list of DataFrames.
    flattened = [item for sublist in segments_diffstd for item in sublist]
    # Next, concatenate the DataFrames together.
    df = pd.concat(flattened)

    return (df, tests_run, diagnostics)

def generate_segment_means(series_segments, num_series, num_segments):
    means = []
    for j in range(num_segments):
        C = [series_segments[i][j]['value'] for i in range(num_series)]
        means.append([sum(x)/num_series for x in zip(*C)])
    return means

def diffstd(s1v, s2v):
    # Find the differences between the two input segments.
    dt = [x1 - x2 for (x1, x2) in zip(s1v, s2v)]
    n = len(s1v)
    mu = np.mean(dt)
    # For each difference, square its distance from the mean.  This guarantees all numbers are positive.
    diff2 = [(d-mu)**2 for d in dt]
    # Sum the squared differences, divide by the number of data points (to get an average),
    # and take the square root of the result.  This returns a single number, the DIFFSTD comparing
    # these two segments.
    return (np.sum(diff2)/n)**0.5

def check_diffstd(series_segments, segment_means, num_series, num_segments):
    # For each series, make a pairwise comparison against the average.
    for i in range(num_series):
        for j in range(num_segments):
            series_segments[i][j]['segment_number'] = j
            series_segments[i][j]['diffstd_distance'] = diffstd(series_segments[i][j]['value'], segment_means[j])
    return series_segments

def score_results(df, tests_run, sensitivity_score):
    # Calculate anomaly score for each series independently.
    # This is because DIFFSTD distances are not normalized across series.
    series = [y for x, y in df.groupby("series_key", as_index=False)]
    num_series = len(series)
    diagnostics = { }

    for i in range(num_series):
        # DIFFSTD doesn't have a hard cutoff point describing when something is (or is not) an outlier.
        # Therefore, to reduce the number of results, we'll start with 1.5 * mean of diffstd distances as a max distance score.
        diffstd_mean = series[i]['diffstd_distance'].mean()

        # Subtract from 1.5 the sensitivity_score/100.0, so at 100 sensitivity, we use 0.5 * mean as a max distance from the mean.
        # Ex:  if the mean is 10 and sensitivity_score is 0, we'll look for segments with DIFFSTD above (10 + 1.5*10) = 25
        # With sensitivity_score 100, the cutoff score will be 15.
        diffstd_sensitivity_threshold = diffstd_mean + ((1.5 - (sensitivity_score / 100.0)) * diffstd_mean)

        # The diffstd_score is the percentage difference between the distance and the sensitivity threshold.
        series[i]['diffstd_score'] = (series[i]['diffstd_distance'] - diffstd_sensitivity_threshold) / diffstd_sensitivity_threshold

        # Our anomaly score is the diffstd_score.
        series[i]['anomaly_score'] = series[i]['diffstd_score']

        diagnostics["Series " + str(i)] = {
            "Mean DIFFSTD distance": diffstd_mean,
            "DIFFSTD sensitivity threshold": diffstd_sensitivity_threshold
        }

    return (pd.concat(series), diagnostics)

def determine_outliers(
    df,
    max_fraction_anomalies
):
    series = [y for x, y in df.groupby("series_key", as_index=False)]

    # Get the 100-Nth percentile of anomaly score.
    # Ex:  if max_fraction_anomalies = 0.1, get the
    # 90th percentile anomaly score.
    max_fraction_anomaly_scores = [np.quantile(s['anomaly_score'], 1.0 - max_fraction_anomalies) for s in series]
    diagnostics = {"Max fraction anomaly scores":  max_fraction_anomaly_scores }

    # When scoring outliers, we made 0.01 the sensitivity threshold, as 0 means no differences.
    # If the max fraction anomaly score is greater than 0, it means that we have MORE outliers
    # than our max_fraction_anomalies supports, and therefore we
    # need to cut it off before we get down to our sensitivity score.
    # Otherwise, sensitivity score stays the same and we operate as normal.
    sensitivity_thresholds = [max(0.01, mfa) for mfa in max_fraction_anomaly_scores]
    diagnostics["Sensitivity scores"] = sensitivity_thresholds

    # We treat segments as outliers, not individual data points.  Mark each segment with a sufficiently large
    # anomaly score as an outlier for subsequent review.
    for i in range(len(series)):
        series[i]['is_anomaly'] = [score >= sensitivity_thresholds[i] for score in series[i]['anomaly_score']]

    return (pd.concat(series), diagnostics)
