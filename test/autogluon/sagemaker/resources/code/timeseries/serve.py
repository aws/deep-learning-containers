"""Inference handler packaged with the trained AutoGluon time series model."""

from io import StringIO

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def model_fn(model_dir):
    predictor = TimeSeriesPredictor.load(model_dir)
    predictor.persist()
    return predictor


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    if input_content_type != "application/json":
        raise ValueError(f"unsupported input content type: {input_content_type}")
    if output_content_type != "application/json":
        raise ValueError(f"unsupported output content type: {output_content_type}")

    data = TimeSeriesDataFrame.from_data_frame(
        pd.read_json(StringIO(request_body)),
        id_column="item_id",
        timestamp_column="timestamp",
    )
    predictions = model.predict(data).reset_index()
    return predictions.to_json(orient="records", date_format="iso"), "application/json"
