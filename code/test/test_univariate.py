from src.app.models.univariate import *
import pandas as pd
import pytest

def test_detect_univariate_statistical_returns_correct_number_of_rows_single():
    # Arrange
    df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sensitivity_score = 75
    max_fraction_anomalies = 0.20
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    # Assert:  the DataFrame is the same length
    # Comment out the following assert and uncomment the next assert if you want to see an example of a failure.
    assert(df_out.shape[0] == df.shape[0])
    #assert(df_out.shape[0] == 4)

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
