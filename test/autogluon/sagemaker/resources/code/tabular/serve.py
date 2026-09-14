"""Inference handler packaged with the trained AutoGluon tabular model."""

from io import StringIO

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    predictor = TabularPredictor.load(model_dir)
    predictor.persist()
    return predictor


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    if input_content_type != "application/json":
        raise ValueError(f"unsupported input content type: {input_content_type}")
    if output_content_type != "application/json":
        raise ValueError(f"unsupported output content type: {output_content_type}")

    data = pd.read_json(StringIO(request_body))
    predictions = model.predict(data, as_pandas=True)
    return predictions.to_frame(name=model.label).to_json(orient="records"), "application/json"
