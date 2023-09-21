# Finding Ghosts in Your Data
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import datetime
from app.models import univariate

app = FastAPI()

@app.get("/")
def doc():
    return {
        "message": "Welcome to the anomaly detector service, based on the book Finding Ghosts in Your Data!",
        "documentation": "If you want to see the OpenAPI specification, navigate to the /redoc/ path on this server."
    }

# Univariate statistical anomaly detection
# For more information on this, review chapters 6-8
class Univariate_Statistical_Input(BaseModel):
    key: str
    value: float

    
@app.post("/detect/univariate")
def post_univariate(
    input_data: List[Univariate_Statistical_Input],
    sensitivity_score: float = 50,
    max_fraction_anomalies: float = 1.0,
    debug: bool = False
):
    df = pd.DataFrame(i.__dict__ for i in input_data)

    (df, weights, details) = univariate.detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    
    # If debug = False, include only key, value, is_anomaly, and anomaly_score.  Remove other values
    results = { "anomalies": json.loads(df.to_json(orient='records')) }
    
    if (debug):
        results.update({ "debug_weights": weights })
        results.update({ "debug_details": details })
    return results
