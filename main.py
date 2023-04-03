from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd

from data_model import DataModel
from prediction_model import PredictionModel


app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def hello_world(request: Request):
   return templates.TemplateResponse("index.html", { "request": request })

@app.post("/predict")
def make_predictions(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])

    predicion_model = PredictionModel()
    results = predicion_model.make_predictions(df)
    return results.tolist()
