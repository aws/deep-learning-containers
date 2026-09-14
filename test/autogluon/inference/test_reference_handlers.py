"""Compatibility checks for AutoGluon Cloud serving handlers."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

RESOURCE_DIR = Path(__file__).parent / "resources"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timeseries_handler_contract(monkeypatch):
    class Predictor:
        target = "target"
        _id_column = "item_id"
        _timestamp_column = "timestamp"

        def predict(self, data, known_covariates=None):
            assert data == "past"
            assert known_covariates == "future"
            return "forecast"

    timeseries = _module("autogluon.timeseries", TimeSeriesPredictor=object)
    serving_utils = _module("serving_utils")
    serving_utils_timeseries = _module(
        "serving_utils.timeseries",
        parse_payload=lambda *args, **kwargs: ("past", "future", {}),
        render_response=lambda predictions, accept: (predictions.encode(), accept),
    )
    monkeypatch.setitem(sys.modules, "autogluon", _module("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.timeseries", timeseries)
    monkeypatch.setitem(sys.modules, "serving_utils", serving_utils)
    monkeypatch.setitem(sys.modules, "serving_utils.timeseries", serving_utils_timeseries)

    handler = _load(RESOURCE_DIR / "timeseries_serve.py", "timeseries_serve")

    assert handler.transform_fn(Predictor(), "request", "application/json") == (
        b"forecast",
        "application/json",
    )


def test_multimodal_handler_contract(monkeypatch):
    class Learner:
        def __init__(self):
            self._label_column = "label"
            self._column_types = {"feature": "text", "label": "categorical"}

    class Predictor:
        def __init__(self):
            self._learner = Learner()

        @classmethod
        def load(cls, model_dir):
            assert model_dir == "/opt/ml/model"
            return cls()

    monkeypatch.setitem(sys.modules, "numpy", _module("numpy"))
    monkeypatch.setitem(sys.modules, "pandas", _module("pandas", Series=object))
    monkeypatch.setitem(sys.modules, "autogluon", _module("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.core", _module("autogluon.core"))
    monkeypatch.setitem(
        sys.modules,
        "autogluon.core.constants",
        _module("autogluon.core.constants", BINARY="binary", MULTICLASS="multiclass"),
    )
    monkeypatch.setitem(
        sys.modules,
        "autogluon.core.utils",
        _module("autogluon.core.utils", get_pred_from_proba_df=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "autogluon.multimodal",
        _module("autogluon.multimodal", MultiModalPredictor=Predictor),
    )

    handler = _load(RESOURCE_DIR / "multimodal_serve.py", "multimodal_serve")

    assert handler.model_fn("/opt/ml/model")._learner._label_column == "label"
    with pytest.raises(ValueError, match="unsupported input content type"):
        handler.transform_fn(Predictor(), b"request", "unsupported")
