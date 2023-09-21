# Finding Ghosts in Your Data
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import datetime

app = FastAPI()

@app.get("/")
def doc():
    return {
        "message": "Welcome to the anomaly detector service, based on the book Finding Ghosts in Your Data!",
        "documentation": "If you want to see the OpenAPI specification, navigate to the /redoc/ path on this server."
    }
