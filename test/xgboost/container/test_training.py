"""Training container tests — rewritten from SMFrameworksXGBoost3_0-5Tests.

Covers:
- Valid training: libsvm, csv, single/multi file, weights, HPO metrics, objectives,
  verbosity, checkpoint/reload for spot instances
- Invalid training: missing data, wrong content types, invalid hyperparameters,
  pipe mode
"""

import copy
import json
import os
import re

import pytest

from .container_helper import run_training, run_distributed_training

# ---------------------------------------------------------------------------
# Standard configs (mirrors configs.py from reference tests)
# ---------------------------------------------------------------------------

STD_HP = {
    "eval_metric": "error",
    "predictor": "cpu_predictor",
    "nthread": "8",
    "sketch_eps": "0.03",
    "base_score": "0.5",
    "scale_pos_weight": "1.0",
    "tree_method": "auto",
    "normalize_type": "tree",
    "max_depth": "6",
    "sample_type": "uniform",
    "booster": "gbtree",
    "objective": "binary:logistic",
    "rate_drop": "0.0",
    "updater": "grow_colmaker,prune",
    "lambda": "1.0",
    "eta": "0.3",
    "alpha": "0.0",
    "process_type": "default",
    "dsplit": "row",
    "max_delta_step": "0",
    "min_child_weight": "1.0",
    "colsample_bytree": "1.0",
    "max_leaves": "0",
    "lambda_bias": "0.0",
    "grow_policy": "depthwise",
    "tweedie_variance_power": "1.5",
    "max_bin": "256",
    "refresh_leaf": "1",
    "num_round": "10",
    "early_stopping_rounds": "5",
    "colsample_bylevel": "1",
    "one_drop": "0",
    "subsample": "1.0",
    "skip_drop": "0.0",
    "gamma": "0.0",
}

STD_IDC = {
    "train": {
        "ContentType": "libsvm",
        "S3DistributionType": "FullyReplicated",
        "TrainingInputMode": "File",
    },
    "validation": {
        "ContentType": "libsvm",
        "S3DistributionType": "FullyReplicated",
        "TrainingInputMode": "File",
    },
}

STD_RC = {"current_host": "algo-1", "hosts": ["algo-1"]}

STD_CPC = {"LocalPath": "/opt/ml/checkpoints"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _libsvm_dir(resources):
    return os.path.join(resources, "data", "single-libsvm")


def _csv_dir(resources):
    return os.path.join(resources, "data", "single-csv")


def _run(docker_client, image_uri, resources, hp, idc, rc, train_files,
         val_files=None, cpc=None, env=None):
    return run_training(
        docker_client, image_uri, hp, idc, rc,
        training_files=train_files,
        validation_files=val_files,
        checkpointconfig=cpc,
        environment=env,
    )


def _assert_success(result, regex=None):
    exit_code, logs, model_files, _ = result
    assert exit_code == 0, f"Training failed:\n{logs}"
    assert len(model_files) == 1, f"Expected 1 model file, got {model_files}"
    if regex:
        assert re.search(regex, logs), f"Pattern {regex!r} not found in logs"


def _assert_failed(result, regex="UserError:"):
    exit_code, logs, _, _ = result
    assert re.search(regex, logs), f"Pattern {regex!r} not found in logs"


# ===========================================================================
# Valid training tests
# ===========================================================================

class TestValidTraining:

    def test_single_file_libsvm(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/libsvm"
        idc["validation"]["ContentType"] = "libsvm"
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "agaricus.libsvm.train")],
                      [os.path.join(d, "agaricus.libsvm.test")])
        _assert_success(result)

    def test_single_file_libsvm_weights(self, docker_client, image_uri, training_resources):
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      [os.path.join(d, "agaricus.libsvm.train.weights")],
                      [os.path.join(d, "agaricus.libsvm.test")])
        _assert_success(result)

    def test_single_file_libsvm_hpo_param(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        d = _libsvm_dir(training_resources)
        for metric in ["validation:rmse", "validation:mae", "validation:logloss",
                       "validation:error", "validation:auc", "validation:aucpr",
                       "validation:ndcg", "validation:map", "validation:accuracy",
                       "validation:f1", "validation:mse"]:
            hp["_tuning_objective_metric"] = metric
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          [os.path.join(d, "agaricus.libsvm.train")],
                          [os.path.join(d, "agaricus.libsvm.test")])
            _assert_success(result, regex=metric.replace(":", "-"))

    def test_single_file_libsvm_multiclass_hpo(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["objective"] = "multi:softmax"
        hp["num_class"] = 3
        hp["eval_metric"] = "merror"
        hp["_tuning_objective_metric"] = "validation:merror"
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                      [os.path.join(d, "synthetic_multi.libsvm.train")],
                      [os.path.join(d, "synthetic_multi.libsvm.train")])
        _assert_success(result, regex="validation-merror")

    def test_single_file_libsvm_hpo_param_non_overlapping(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["_tuning_objective_metric"] = "validation:logloss"
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                      [os.path.join(d, "agaricus.libsvm.train")],
                      [os.path.join(d, "agaricus.libsvm.test")])
        _assert_success(result, regex="(?=.*validation-logloss:.*)(?=.*validation-error:.*)")

    def test_single_file_output_both_default_and_custom_metrics(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        eval_metrics = ["logloss", "f1", "error"]
        hp["eval_metric"] = ",".join(eval_metrics)
        for hpo_metric in ["validation:accuracy", "validation:mae"]:
            hp["_tuning_objective_metric"] = hpo_metric
            d = _libsvm_dir(training_resources)
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          [os.path.join(d, "agaricus.libsvm.train")],
                          [os.path.join(d, "agaricus.libsvm.test")])
            all_metrics = list(set(eval_metrics) | {hpo_metric})
            regex = "".join(f"(?=.*{m.replace(':', '-')})" for m in all_metrics)
            _assert_success(result, regex=regex)

    def test_single_file_libsvm_iterate_objectives(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        d = _libsvm_dir(training_resources)
        for obj in ["reg:squarederror", "reg:logistic", "binary:logistic",
                     "binary:logitraw", "count:poisson"]:
            hp["objective"] = obj
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          [os.path.join(d, "agaricus.libsvm.train")],
                          [os.path.join(d, "agaricus.libsvm.test")])
            _assert_success(result)

    def test_single_file_libsvm_threshold_eval_metric(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["eval_metric"] = "error@0.8"
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                      [os.path.join(d, "agaricus.libsvm.train")],
                      [os.path.join(d, "agaricus.libsvm.test")])
        _assert_success(result)

    def test_single_file_libsvm_verbosity(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["verbosity"] = "3"
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                      [os.path.join(d, "agaricus.libsvm.train")],
                      [os.path.join(d, "agaricus.libsvm.test")])
        _assert_success(result)

    def test_multi_files_libsvm(self, docker_client, image_uri, training_resources):
        d = os.path.join(training_resources, "data", "multi-libsvm")
        train_dir = os.path.join(d, "train")
        val_dir = os.path.join(d, "val")
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      [train_dir], [val_dir])
        _assert_success(result)

    def test_single_file_csv(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/csv"
        idc["validation"]["ContentType"] = "csv"
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train.csv")],
                      [os.path.join(d, "val.csv")])
        _assert_success(result)

    def test_single_file_csv_weights(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/csv"
        idc["validation"]["ContentType"] = "text/csv"
        hp = copy.deepcopy(STD_HP)
        hp["csv_weights"] = "1"
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, idc, STD_RC,
                      [os.path.join(d, "train.csv.weights")],
                      [os.path.join(d, "val.csv")])
        _assert_success(result)

    def test_multi_file_csv(self, docker_client, image_uri, training_resources):
        d = os.path.join(training_resources, "data", "multi-csv")
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "csv"
        idc["validation"]["ContentType"] = "csv"
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train")],
                      [os.path.join(d, "val")])
        _assert_success(result)

    def test_single_file_csv_space_separated(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "csv"
        idc.pop("validation", None)
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train_space.csv")])
        _assert_success(result)

    def test_single_file_csv_sci_notation(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "csv"
        idc.pop("validation", None)
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train_sci.csv")])
        _assert_success(result)

    def test_single_file_csv_empty_cells(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "csv"
        idc.pop("validation", None)
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train_empty_cell.csv")])
        _assert_success(result)

    def test_two_container_with_libsvm_data(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/libsvm"
        idc["validation"]["ContentType"] = "text/libsvm"
        d = _libsvm_dir(training_resources)
        train_files = [os.path.join(d, "agaricus.libsvm.train")]
        hosts = ["algo-1", "algo-2"]
        rcs = [
            {"current_host": "algo-1", "hosts": hosts},
            {"current_host": "algo-2", "hosts": hosts},
        ]
        results = run_distributed_training(
            docker_client, image_uri, STD_HP, idc, rcs, train_files,
        )
        assert results[0][0] == 0, f"Container 1 failed:\n{results[0][1]}"
        assert results[1][0] == 0, f"Container 2 failed:\n{results[1][1]}"
        model_files = [f for f in os.listdir(results[0][2]["model"]) if "model" in f]
        assert len(model_files) == 1

    def test_two_container_with_libsvm_data_shardedbykey(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/libsvm"
        idc["train"]["S3DistributionType"] = "ShardedByS3Key"
        idc["validation"]["ContentType"] = "text/libsvm"
        idc["validation"]["S3DistributionType"] = "ShardedByS3Key"
        d = _libsvm_dir(training_resources)
        train_files = [os.path.join(d, "agaricus.libsvm.train")]
        hosts = ["algo-1", "algo-2"]
        rcs = [
            {"current_host": "algo-1", "hosts": hosts},
            {"current_host": "algo-2", "hosts": hosts},
        ]
        results = run_distributed_training(
            docker_client, image_uri, STD_HP, idc, rcs, train_files,
        )
        assert results[0][0] == 0, f"Container 1 failed:\n{results[0][1]}"
        assert results[1][0] == 0, f"Container 2 failed:\n{results[1][1]}"
        model_files = [f for f in os.listdir(results[0][2]["model"]) if "model" in f]
        assert len(model_files) == 1

    def test_checkpoint_and_reload(self, docker_client, image_uri, training_resources):
        """Train 10 rounds, verify checkpoints, then resume to 20 rounds."""
        hp1 = copy.deepcopy(STD_HP)
        hp1["num_round"] = 10
        hp1["eval_metric"] = "error"
        hp1.pop("early_stopping_rounds", None)

        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/libsvm"
        idc.pop("validation", None)

        d = _libsvm_dir(training_resources)
        train_files = [os.path.join(d, "agaricus.libsvm.train")]

        # Phase 1: train 10 rounds
        exit_code, logs, model_files, paths = run_training(
            docker_client, image_uri, hp1, idc, STD_RC,
            training_files=train_files, checkpointconfig=STD_CPC,
        )
        assert exit_code == 0
        assert len(model_files) == 1

        ckpt_files = os.listdir(paths["checkpoints"])
        assert len(ckpt_files) >= 1, f"No checkpoint files found"
        regex = r"\[\d+\].*(?=.*train-error:.*)"
        assert len(re.findall(regex, logs)) == 10

        # Phase 2: resume to 20 rounds using same opt_ml dir
        hp2 = copy.deepcopy(STD_HP)
        hp2["num_round"] = 20
        hp2["eval_metric"] = "error"
        hp2.pop("early_stopping_rounds", None)

        config_dir = paths["input_config"]
        with open(os.path.join(config_dir, "hyperparameters.json"), "w") as f:
            json.dump(hp2, f)

        # Clear model dir for fresh output
        for mf in os.listdir(paths["model"]):
            os.remove(os.path.join(paths["model"], mf))

        tmpdir = paths["input_config"].rsplit("/input/", 1)[0]
        volumes = {tmpdir: {"bind": "/opt/ml", "mode": "rw"}}

        container = docker_client.containers.run(
            image_uri, command="train", volumes=volumes,
            detach=True,
        )
        try:
            result = container.wait(timeout=300)
            exit_code2 = result.get("StatusCode", -1)
        except Exception:
            exit_code2 = -1
        finally:
            logs2 = container.logs().decode("utf-8", errors="replace")
            container.remove(force=True)

        assert exit_code2 == 0
        ckpt_files2 = os.listdir(paths["checkpoints"])
        assert len(ckpt_files2) >= 1
        assert len(re.findall(regex, logs2)) >= 10


# ===========================================================================
# Invalid training tests
# ===========================================================================

class TestInvalidTraining:

    def _get_libsvm_data(self, resources, with_validation=True):
        d = _libsvm_dir(resources)
        train = [os.path.join(d, "agaricus.libsvm.train")]
        val = [os.path.join(d, "agaricus.libsvm.test")]
        return (train, val) if with_validation else train

    def test_no_training_data(self, docker_client, image_uri, training_resources):
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC, [])
        _assert_failed(result)

    def test_no_validation_data(self, docker_client, image_uri, training_resources):
        train = self._get_libsvm_data(training_resources, False)
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      train, [])
        _assert_failed(result)

    def test_invalid_data_csv_content_type(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "csv"
        idc["validation"]["ContentType"] = "csv"
        d = os.path.join(training_resources, "data", "invalid-data")
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "data.rec")], [os.path.join(d, "data.rec")])
        _assert_failed(result)

    def test_csv_alpha_with_csv_content_type(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["ContentType"] = "text/csv"
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      [os.path.join(d, "train_alpha.csv")])
        _assert_failed(result)

    def test_csv_data_with_libsvm_content_type(self, docker_client, image_uri, training_resources):
        d = _csv_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      [os.path.join(d, "train.csv")], [os.path.join(d, "val.csv")])
        _assert_failed(result, regex="UserError:")

    def test_invalid_data_with_libsvm_content_type(self, docker_client, image_uri, training_resources):
        d = os.path.join(training_resources, "data", "invalid-data")
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      [os.path.join(d, "data.rec")], [os.path.join(d, "data.rec")])
        _assert_failed(result)

    @pytest.mark.parametrize("param,values", [
        ("eta", ["-0.1", "1.01", "invalid_string"]),
        ("gamma", ["-0.1", "invalid_string"]),
        ("max_depth", ["-0.1", "invalid_string"]),
        ("min_child_weight", ["-0.1", "invalid_string"]),
        ("max_delta_step", ["-0.1", "invalid_string"]),
        ("colsample_bytree", ["-0.1", "0", "invalid_string"]),
        ("colsample_bylevel", ["-0.1", "0", "invalid_string"]),
        ("tree_method", ["invalid_method", "gpu_exact"]),
        ("sketch_eps", ["0", "1", "invalid_string"]),
        ("refresh_leaf", ["invalid", "2"]),
        ("process_type", ["invalid", "0.01"]),
        ("grow_policy", ["invalid", "0.01"]),
        ("sample_type", ["invalid", "0.01"]),
        ("normalize_type", ["invalid", "0.01"]),
        ("rate_drop", ["invalid", "-0.01", "1.01"]),
        ("one_drop", ["invalid", "-0.01", "1.01"]),
        ("skip_drop", ["invalid", "-0.01", "1.01"]),
        ("tweedie_variance_power", ["invalid", "1", "2"]),
        ("eval_metric", ["invalid", "1", "rmse,invalid", "error@nonfloat"]),
        ("booster", ["invalid", "1"]),
        ("verbosity", ["invalid", "-1", "4", "0.5"]),
    ])
    def test_invalid_hyperparameter(self, docker_client, image_uri, training_resources,
                                    param, values):
        train, val = self._get_libsvm_data(training_resources)
        hp = copy.deepcopy(STD_HP)
        for v in values:
            hp[param] = v
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          train, val)
            _assert_failed(result)

    def test_missing_num_round(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp.pop("num_round", None)
        train, val = self._get_libsvm_data(training_resources)
        result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                      train, val)
        _assert_failed(result)

    def test_multiclass_without_num_class(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        train, val = self._get_libsvm_data(training_resources)
        for obj in ["multi:softmax", "multi:softprob"]:
            hp["objective"] = obj
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          train, val)
            _assert_failed(result)

    def test_libsvm_data_alpha_with_libsvm_content_type(self, docker_client, image_uri, training_resources):
        d = _libsvm_dir(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, STD_IDC, STD_RC,
                      [os.path.join(d, "agaricus.alpha.train")],
                      [os.path.join(d, "agaricus.alpha.train")])
        _assert_failed(result)

    def test_invalid_updater_for_update_process_type(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["process_type"] = "update"
        train = self._get_libsvm_data(training_resources, False)
        idc = copy.deepcopy(STD_IDC)
        idc.pop("validation", None)
        result = _run(docker_client, image_uri, training_resources, hp, idc, STD_RC, train)
        _assert_failed(result)

        hp["updater"] = "refresh,invalid"
        result = _run(docker_client, image_uri, training_resources, hp, idc, STD_RC, train)
        _assert_failed(result)

    def test_invalid_updater_for_gblinear(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["booster"] = "gblinear"
        train = self._get_libsvm_data(training_resources, False)
        idc = copy.deepcopy(STD_IDC)
        idc.pop("validation", None)
        result = _run(docker_client, image_uri, training_resources, hp, idc, STD_RC, train)
        _assert_failed(result)

        hp["updater"] = "shotgun,grow_colmaker"
        result = _run(docker_client, image_uri, training_resources, hp, idc, STD_RC, train)
        _assert_failed(result)

    def test_auc_with_invalid_objective(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        hp["eval_metric"] = "auc"
        train, val = self._get_libsvm_data(training_resources)
        for obj in ["reg:squarederror", "reg:linear", "reg:gamma"]:
            hp["objective"] = obj
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          train, val)
            _assert_failed(result)

    def test_invalid_eval_metric_values(self, docker_client, image_uri, training_resources):
        hp = copy.deepcopy(STD_HP)
        train, val = self._get_libsvm_data(training_resources)
        for invalid in ["<function", "auc@0.5"]:
            hp["eval_metric"] = invalid
            result = _run(docker_client, image_uri, training_resources, hp, STD_IDC, STD_RC,
                          train, val)
            _assert_failed(result)

    def test_pipe_mode_rejected(self, docker_client, image_uri, training_resources):
        idc = copy.deepcopy(STD_IDC)
        idc["train"]["TrainingInputMode"] = "Pipe"
        train, val = self._get_libsvm_data(training_resources)
        result = _run(docker_client, image_uri, training_resources, STD_HP, idc, STD_RC,
                      train, val)
        _assert_failed(result)
