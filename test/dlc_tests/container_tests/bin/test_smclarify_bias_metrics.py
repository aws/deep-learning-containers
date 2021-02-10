#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import sys
import json
import boto3

from typing import Dict, Optional

import pandas as pd
from smclarify.bias.report import FacetColumn, LabelColumn, bias_report, StageType
from smclarify.util.dataset import Datasets

logger = logging.getLogger(__name__)

def fetch_input_data() -> pd.DataFrame:
    dataset = Datasets()
    s3_input_obj = dataset("german_csv")
    df = s3_input_obj.read_csv_data()
    return df


def get_expected_results() -> Dict:
    s3_client = boto3.client("s3")
    test_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "bias_metrics_results.json"
    s3_client.download_file("sagemaker-clarify-datasets",
                            f"statlog/result/{file_name}",
                            f"{test_dir}/{file_name}")
    results_file = os.path.join(test_dir, file_name)
    with open(results_file) as json_file:
        expected_results = json.load(json_file)
    return expected_results


def get_predicted_labels() -> pd.DataFrame:
    dataset = Datasets()
    s3_pred_label_obj = dataset("german_predicted_labels")
    predicted_labels = s3_pred_label_obj.read_csv_data(index_col=0)
    return predicted_labels.squeeze()


def get_pretraining_bias_metrics(
    dataframe: pd.DataFrame, facet_column: FacetColumn, label_column: LabelColumn, group_variable: Optional[pd.Series]
) -> Dict:
    # Measure pre-training bias for the ForeignWorker attribute
    return bias_report(
        dataframe,
        facet_column,
        label_column,
        stage_type=StageType.PRE_TRAINING,
        metrics=["all"],
        group_variable=group_variable,
    )


def get_posttraining_bias_metrics(
    dataframe: pd.DataFrame,
    facet_column: FacetColumn,
    label_column: LabelColumn,
    pred_label_column: LabelColumn,
    group_variable: Optional[pd.Series],
) -> Dict:
    # Measure the post-training bias for the ForeignWorker attribute
    report = bias_report(
        dataframe,
        facet_column,
        label_column,
        stage_type=StageType.POST_TRAINING,
        predicted_label_column=pred_label_column,
        metrics=["all"],
        group_variable=group_variable,
    )
    return report


def test_bias_metrics():
    dataframe = fetch_input_data()
    label_data = dataframe.pop("Class1Good2Bad")
    label_column = LabelColumn("Class1Good2Bad", label_data, [1])
    facet_column = FacetColumn("ForeignWorker", [1])
    group_variable = dataframe["A151"]

    # pre_training_bias metrics
    pre_training_metrics = get_pretraining_bias_metrics(dataframe, facet_column, label_column, group_variable)

    # post training bias metrics
    predicted_labels = get_predicted_labels()
    pred_label_column = LabelColumn("_predicted_labels", predicted_labels, [1])

    post_training_metrics = get_posttraining_bias_metrics(
        dataframe, facet_column, label_column, pred_label_column, group_variable
    )

    expected_results = get_expected_results()
    pre_training_expected_result = expected_results.get("pre_training_bias_metrics")
    post_training_expected_result = expected_results.get("post_training_bias_metrics")

    if not (pre_training_metrics == pre_training_expected_result):
        raise AssertionError("Pre_training Bias Metrics values differ from expected Metrics")
    if not (post_training_metrics == post_training_expected_result):
        raise AssertionError("Post_training Bias Metrics values differ from expected Metrics")
    print("Test SMClarify Bias Metrics succeeded!")


if __name__ == "__main__":
    try:
        sys.exit(test_bias_metrics())
    except KeyboardInterrupt:
        pass
