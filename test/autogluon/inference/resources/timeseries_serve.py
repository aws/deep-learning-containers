# Adapted from AutoGluon Cloud's Apache-2.0 licensed timeseries_serve.py.
import json
import os
import shutil

from autogluon.timeseries import TimeSeriesPredictor
from serving_utils.timeseries import parse_payload, render_response


def model_fn(model_dir):
    """Load a predictor from a writable copy of the SageMaker model directory."""
    tmp_model_dir = os.path.join("/tmp", "model")
    try:
        shutil.copytree(model_dir, tmp_model_dir, dirs_exist_ok=False)
    except FileExistsError:
        pass
    model = TimeSeriesPredictor.load(tmp_model_dir)
    if hasattr(model, "persist"):
        model.persist()

    metadata_path = os.path.join(tmp_model_dir, "predictor_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        model._id_column = metadata["id_column"]
        model._timestamp_column = metadata["timestamp_column"]
    else:
        model._id_column = "item_id"
        model._timestamp_column = "timestamp"
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    time_series, known_covariates, _ = parse_payload(
        request_body,
        input_content_type,
        id_column=model._id_column,
        timestamp_column=model._timestamp_column,
        target_column=model.target,
    )
    predictions = model.predict(time_series, known_covariates=known_covariates)
    return render_response(predictions, output_content_type)
