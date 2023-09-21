from src.app.models.univariate import *
import pandas as pd
import pytest

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1],
    [1, 2, 3, 4.5, 6.78, 9.10],
    [],
    [1000, 1500, 2230, 13, 1780, 1629, 2202, 2025]
])
def test_detect_univariate_statistical_returns_correct_number_of_rows(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensitivity_score = 50
    max_fraction_anomalies = 0.20
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    # Assert:  the DataFrame is the same length
    assert(df_out.shape[0] == df.shape[0])

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90],
    [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, -13],
    [0.01, 0.03, 0.05, 0.02, 0.01, 0.03, 0.40],
    [1000, 1500, 1230, 13, 1780, 1629, 1450, 1106],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19.4]
])
def test_detect_univariate_statistical_returns_single_anomaly(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensitivity_score = 50
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly one anomaly
    assert(num_anomalies == 1)

@pytest.mark.parametrize("df_input", [
    [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 2550, 9000],
    [1, 1, 1, 2, 3, 3, 5, -13],
    [1, 1, 1, 2, 3, 5, 5, 5, -13, 18],
    [0.01, 0.03, 2, 0.02, 0.01, 0.03, -2.8],
    [1000, 1250, 1173, 13, 1306, 1222, 1064, 1071, 6]
])
def test_detect_univariate_statistical_returns_two_anomalies(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensitivity_score = 50
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly two anomalies
    assert(num_anomalies == 2)

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1000, 1500, 2230, 13, 1780, 1629, 3202, 3025, 6]
])
def test_detect_univariate_statistical_returns_zero_anomalies(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensitivity_score = 30
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly zero anomalies
    assert(num_anomalies == 0)

anomalous_sample = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 2550, 9000]
@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (anomalous_sample, 100, 17),
    (anomalous_sample, 95, 15),# In chapter 6, this is 15; in chapter 7, it decreases to 12; in chapter 9, it increases again to 13.
    (anomalous_sample, 85, 8), # In chapter 6, this is 8; in chapter 7, it decreases to 5; in chapter 9, it increases again to 9.
    (anomalous_sample, 75, 5), # In chapter 6, this is 5; in chapter 7, it decreases to 2; in chapter 9, it increases again to 6.
    (anomalous_sample, 50, 2),
    (anomalous_sample, 25, 2),
    (anomalous_sample, 1, 1)   # In chapter 6, this is 1; in chapter 7, it decreases to 0; in chapter 9, it increases back to 1.
])
def test_detect_univariate_statistical_sensitivity_affects_anomaly_count(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    max_fraction_anomalies = 1.0
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have the correct number of anomalies
    assert(num_anomalies == number_of_anomalies)

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (anomalous_sample, 0.0, 0),
    (anomalous_sample, 0.01, 1),
    (anomalous_sample, 0.1, 2),
    (anomalous_sample, 0.2, 4), # In chapter 7, this is 3; in chapter 9, it increases to 4.
    (anomalous_sample, 0.3, 6), # In chapter 7, this is 5; in chapter 9, it decreases to 4 without the equivalency change and increases to 6 with it.
    (anomalous_sample, 0.4, 7), # In chapter 7, this is 6; in chapter 9, it increases to 7.
    (anomalous_sample, 0.5, 9), # In chapter 7, this is 8; in chapter 9, it decreases to 7 without the equivalency change and increases to 9 with it.
    (anomalous_sample, 0.6, 10), # In chapter 7, this is 9; in chapter 9, it increases to 10.
    (anomalous_sample, 0.7, 12),
    (anomalous_sample, 0.8, 13), # In chapter 7, this is 12; in chapter 9, it increases to 13.
    (anomalous_sample, 0.9, 15),
    (anomalous_sample, 1.0, 17)
])
def test_detect_univariate_statistical_max_fraction_anomalies_affects_anomaly_count(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["value"])
    sensitivity_score = 100.0
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have the correct number of anomalies
    assert(num_anomalies == number_of_anomalies)
