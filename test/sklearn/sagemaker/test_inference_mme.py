"""Inference tests — both single-model and multi-model endpoints against the
sklearn MME model tarballs. Verifies both endpoint modes round-trip a
CSV payload.
"""

import math

import pytest

from .conftest import (
    data_uri,
    delete_endpoint,
    deploy_endpoint,
    deploy_multi_model_endpoint,
    predict_and_log,
)

# Model tarballs live under a version-slugged prefix so 1.4.2 and 1.9.0 fixtures
# stay isolated (pickle format can drift across sklearn versions).
# MME lazy-loads only the tarball named in `target_model`, so the sibling
# `code/user_code.tar.gz` under this prefix is never invoked as a model.
MME_TARGET_MODELS = ["mme_model_0.tar.gz", "mme_model_1.tar.gz"]
MME_CODE_KEY = "mme_models/code/user_code.tar.gz"

# Baseline predictions on SAMPLE_PAYLOAD keyed by sklearn version, matching
# fixtures at s3://amazonai-algorithms-integration-tests/input/scikit-learn/mme_models/<version>/.
# Compared with math.isclose(rel_tol=1e-6) — not byte-exact. Regenerate with
# test/sklearn/scripts/regen_mme_fixtures.py and add a new entry per sklearn
# version — do not overwrite existing entries.
BASELINE_PREDICTIONS = {
    "1.9.0": {
        "mme_model_0.tar.gz": [0.4714299835128364, 0.35936579802301344],
        "mme_model_1.tar.gz": [0.5477832620234803, 0.35375510160817975],
    },
}

# Fitted models have n_features_in_ = 6. Multi-row CSV exercises batch predict.
SAMPLE_PAYLOAD = "\n".join(
    [
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
        "0.0, 2.0, 4.0, 6.0, 9.0, 3.0",
    ]
)


@pytest.fixture(scope="module")
def mme_env():
    return {
        "SAGEMAKER_PROGRAM": "script.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": data_uri(MME_CODE_KEY),
    }


def _parse_csv_response(response):
    """Parse MMS CSV response (bytes or str) into a list of floats."""
    text = response.decode() if isinstance(response, bytes) else response
    return [float(x) for x in text.strip().splitlines() if x.strip()]


def _assert_predictions(actual, target_model, sklearn_version):
    """Compare endpoint response to known-good predictions for the given
    sklearn version. Falls back to smoke assertion for versions without a
    baseline (e.g. legacy 1.4.2 fixtures that pre-date regen_mme_fixtures.py).
    """
    version_predictions = BASELINE_PREDICTIONS.get(sklearn_version)
    if version_predictions is None:
        assert actual is not None
        return
    expected = version_predictions[target_model]
    parsed = _parse_csv_response(actual)
    assert len(parsed) == len(expected), f"{target_model}: got {parsed}, expected {expected}"
    for p, e in zip(parsed, expected):
        assert math.isclose(p, e, rel_tol=1e-6), f"{target_model}: predicted {p} != expected {e}"


class TestInferenceMME:
    def test_csv_single_model(self, image_uri, role, mme_env, sklearn_version):
        prefix = f"mme_models/{sklearn_version}/"
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri,
                role=role,
                model_data=data_uri(f"{prefix}{MME_TARGET_MODELS[0]}"),
                env=mme_env,
                test_name="inference-single",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            response = predict_and_log(predictor, SAMPLE_PAYLOAD)
            _assert_predictions(response, MME_TARGET_MODELS[0], sklearn_version)
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_csv_multimodel(self, image_uri, role, mme_env, sklearn_version):
        prefix = f"mme_models/{sklearn_version}/"
        endpoint_name = None
        try:
            predictor, endpoint_name, _ = deploy_multi_model_endpoint(
                image_uri=image_uri,
                role=role,
                model_data_prefix=data_uri(prefix),
                env=mme_env,
                test_name="inference-mme",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            for target_model in MME_TARGET_MODELS:
                response = predict_and_log(predictor, SAMPLE_PAYLOAD, target_model=target_model)
                _assert_predictions(response, target_model, sklearn_version)
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
