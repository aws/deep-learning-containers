from djl_python import Input
from djl_python import Output
from autogluon.tabular import TabularPredictor
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.core.constants import REGRESSION
from io import BytesIO, StringIO
import pandas as pd
import os

# Model loading
current_file_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])
model = TabularPredictor.load(current_file_path, require_py_version_match=False)
column_names = model.feature_metadata_in.get_features()

def handle(inputs: Input) -> Output:
    if inputs.is_empty():
        return None
    content_type = inputs.get_property("content-type")
    if content_type == "application/x-parquet":
        data = BytesIO(inputs.get_as_bytes())
        data = pd.read_parquet(data)
    elif content_type == "text/csv":
        data = StringIO(inputs.get_as_string())
        data = pd.read_csv(data)
    elif content_type == "application/json":
        # https://github.com/deepjavalibrary/djl-serving/blob/1ae3a6c157482a49318f38998e9873461609ee99/engines/python/setup/djl_python/inputs.py
        data = inputs.get_as_json()
        data = pd.DataFrame(data)
    else:
        raise ValueError(f"{content_type} input content type not supported.")
    
    if model.problem_type != REGRESSION:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
        pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True)
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()
    
    output = prediction.to_json()
    return Output().add(output).add_property("content-type", "application/json")