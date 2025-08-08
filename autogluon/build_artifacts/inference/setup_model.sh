#!/bin/bash

MODEL_DIR="/opt/ml/model"
# Use /tmp for custom_model since /opt/ml/model is read-only
CUSTOM_MODEL_DIR="/tmp/custom_model"

echo "DEBUG: Contents of ${MODEL_DIR}:"
ls -la "${MODEL_DIR}"

# Check if SageMaker has already extracted the model (which it does automatically)
if [ -f "${MODEL_DIR}/model.py" ] && [ -f "${MODEL_DIR}/learner.pkl" ]; then
    echo "DEBUG: Model already extracted by SageMaker, using existing files"
    # SageMaker has already extracted the model, just use the custom_model directory structure
    rm -rf "${CUSTOM_MODEL_DIR}"
    mkdir -p "${CUSTOM_MODEL_DIR}"
    
    # Copy all model files to custom_model directory
    cp -r "${MODEL_DIR}"/* "${CUSTOM_MODEL_DIR}/" 2>/dev/null || true
    
    echo "DEBUG: Contents of ${CUSTOM_MODEL_DIR} after copying:"
    ls -la "${CUSTOM_MODEL_DIR}"
    
elif [ -f "${MODEL_DIR}/model_1.3.1.tar.gz" ]; then
    echo "DEBUG: Found model_1.3.1.tar.gz, extracting..."
    rm -rf "${CUSTOM_MODEL_DIR}"
    mkdir -p "${CUSTOM_MODEL_DIR}"
    tar -xf "${MODEL_DIR}/model_1.3.1.tar.gz" -C "${CUSTOM_MODEL_DIR}"
    
elif [ -f "${MODEL_DIR}/model.tar.gz" ]; then
    echo "DEBUG: Found model.tar.gz, extracting..."
    rm -rf "${CUSTOM_MODEL_DIR}"
    mkdir -p "${CUSTOM_MODEL_DIR}"
    tar -xf "${MODEL_DIR}/model.tar.gz" -C "${CUSTOM_MODEL_DIR}"
    
else
    echo "ERROR: No model file or extracted model found in ${MODEL_DIR}"
    echo "Looking for: model_1.3.1.tar.gz, model.tar.gz, or extracted model files"
    exit 1
fi

echo "DEBUG: Final contents of ${CUSTOM_MODEL_DIR}:"
ls -la "${CUSTOM_MODEL_DIR}"

# Verify required files exist
if [ ! -f "${CUSTOM_MODEL_DIR}/model.py" ]; then
    echo "ERROR: model.py not found in ${CUSTOM_MODEL_DIR}"
    exit 1
fi

if [ ! -f "${CUSTOM_MODEL_DIR}/learner.pkl" ]; then
    echo "ERROR: learner.pkl not found in ${CUSTOM_MODEL_DIR}"
    exit 1
fi

echo "DEBUG: Setup completed successfully"

# Copy or create model.py
MODEL_PY_SOURCE="/opt/ml/code/model.py"
MODEL_PY_DEST="${CUSTOM_MODEL_DIR}/model.py"

if [ -f "${MODEL_PY_SOURCE}" ]; then
    cp "${MODEL_PY_SOURCE}" "${MODEL_PY_DEST}"
else
    cat > "${MODEL_PY_DEST}" << 'EOF'
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
EOF
fi