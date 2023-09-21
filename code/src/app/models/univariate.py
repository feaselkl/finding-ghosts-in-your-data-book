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
    df_out = df.assign(is_anomaly=False, anomaly_score=0.0)
    return (df_out, [0,0,0], { "message": "No ensemble chosen."})
