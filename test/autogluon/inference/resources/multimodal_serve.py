# Adapted from AutoGluon Cloud's Apache-2.0 licensed multimodal_serve.py.
import base64
import copy
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.multimodal import MultiModalPredictor

column_names = []


def model_fn(model_dir):
    """Load a MultiModalPredictor and retain its inference feature order."""
    model = MultiModalPredictor.load(model_dir)
    if hasattr(model, "_learner") and hasattr(model._learner, "_label_column"):
        label_column = model._learner._label_column
        column_types = copy.copy(model._learner._column_types)
    else:
        label_column = model._label_column
        column_types = copy.copy(model._column_types)
    column_types.pop(label_column)
    globals()["column_names"] = list(column_types.keys())
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    image_bytearrays = None
    if input_content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body))
    elif input_content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
    elif input_content_type == "application/json":
        data = pd.read_json(StringIO(request_body))
    elif input_content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body), orient="records", lines=True)
    elif input_content_type == "application/x-npy":
        image_bytearrays = [base64.b85decode(value) for value in np.load(BytesIO(request_body))]
    elif input_content_type == "application/x-image":
        image_bytearrays = [request_body]
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    if image_bytearrays is not None:
        data = {"image": image_bytearrays}
    elif sorted(data.columns) != sorted(column_names):
        if len(data.columns) != len(column_names):
            raise ValueError(
                f"Invalid data format. Input has {len(data.columns)} columns; "
                f"the model expects {len(column_names)}"
            )
        data.columns = column_names

    if model.problem_type in (BINARY, MULTICLASS):
        probabilities = model.predict_proba(data, as_pandas=True)
        predictions = get_pred_from_proba_df(probabilities, problem_type=model.problem_type)
        probabilities.columns = [f"{column}_proba" for column in probabilities.columns]
        predictions.name = model.label
        output = pd.concat([predictions, probabilities], axis=1)
    else:
        output = model.predict(data, as_pandas=True)
    if isinstance(output, pd.Series):
        output = output.to_frame()

    if "application/x-parquet" in output_content_type:
        return output.to_parquet(), "application/x-parquet"
    if "application/json" in output_content_type:
        return output.to_json(), "application/json"
    if "text/csv" in output_content_type:
        return output.to_csv(index=None), "text/csv"
    raise ValueError(f"{output_content_type} content type not supported")
