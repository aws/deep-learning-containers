"""SageMaker Batch Transform integration test for TF 2.20 inference DLC.

Exercises the CreateTransformJob wire contract and the DLC's request-per-file
behaviour — a code path that the real-time single-model/MME tests don't touch.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest
from test_utils import wait_for_status

LOGGER = logging.getLogger(__name__)

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball
from test_utils import random_suffix_name

# Always CPU — CreateTransformJob wire is device-agnostic, and CI accounts
# have zero TransformJob GPU quota by default.
BATCH_TRANSFORM_INSTANCE_TYPE = "ml.c5.xlarge"


def test_batch_transform_json(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    image_uri,
):
    """End-to-end batch transform on JSON: 3 single-record files, verify 2x output."""
    # Late imports so pytest --collect-only works without the SDK.
    from sagemaker.core.resources import (
        ContainerDefinition,
        Model,
        TransformJob,
    )
    from sagemaker.core.shapes.shapes import (
        TransformDataSource,
        TransformInput,
        TransformOutput,
        TransformResources,
        TransformS3DataSource,
    )

    bucket = sagemaker_session.default_bucket()
    run_id = random_suffix_name("batch", 63)

    with tempfile.TemporaryDirectory(prefix="tf220-batch-") as workdir:
        workdir_path = Path(workdir)

        # 1. Build + upload model.
        tar_path = build_sample_model(output_dir=workdir_path, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/batch/{run_id}/model",
        )

        # 2. Write 3 input files — one JSON body per file (SplitType=None).
        input_dir = workdir_path / "input"
        input_dir.mkdir()
        for i, row in enumerate([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]):
            (input_dir / f"row_{i}.json").write_text(json.dumps({"instances": [row]}))

        s3 = boto_session.client("s3")
        input_prefix = f"tf220-inference-tests/batch/{run_id}/input"
        for f in input_dir.iterdir():
            s3.upload_file(str(f), bucket, f"{input_prefix}/{f.name}")
        s3_input = f"s3://{bucket}/{input_prefix}/"
        s3_output = f"s3://{bucket}/tf220-inference-tests/batch/{run_id}/output/"

        # 3. Create model.
        model_name = random_suffix_name("tf220-batch-model", 63)
        job_name = random_suffix_name("tf220-batch-job", 63)
        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_url=model_data,
            ),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        try:
            # 4. TransformJob has no wait_for_status in SDK v3 — poll refresh().
            job = TransformJob.create(
                transform_job_name=job_name,
                model_name=model_name,
                transform_input=TransformInput(
                    data_source=TransformDataSource(
                        s3_data_source=TransformS3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=s3_input,
                        ),
                    ),
                    content_type="application/json",
                    split_type="None",
                ),
                transform_output=TransformOutput(
                    s3_output_path=s3_output,
                    accept="application/json",
                    assemble_with="None",
                ),
                transform_resources=TransformResources(
                    instance_type=BATCH_TRANSFORM_INSTANCE_TYPE,
                    instance_count=1,
                ),
                session=boto_session,
            )
            def _get_transform_status():
                job.refresh()
                return getattr(job, "transform_job_status", None)

            completed = wait_for_status(
                "Completed",
                wait_periods=90,
                period_length=30,
                get_status_method=_get_transform_status,
            )
            if not completed:
                job.refresh()
                status = getattr(job, "transform_job_status", None)
                assert False, (
                    f"TransformJob failed: status={status!r} "
                    f"failure_reason={getattr(job, 'failure_reason', None)!r}"
                )

            # 5. Download output objects and assert predictions.
            resp = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f"tf220-inference-tests/batch/{run_id}/output/",
            )
            objects = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".out")]
            assert len(objects) == 3, (
                f"expected 3 output objects, found {len(objects)}: {objects!r}"
            )

            expected_by_input = {
                "row_0.json.out": [2.0, 4.0, 6.0],
                "row_1.json.out": [8.0, 10.0, 12.0],
                "row_2.json.out": [14.0, 16.0, 18.0],
            }
            for key in objects:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                parsed = json.loads(body)
                predictions = parsed.get("predictions", [])
                assert predictions, f"empty predictions in {key}: {body!r}"
                first = predictions[0]
                values = first["output"] if isinstance(first, dict) and "output" in first else first
                filename = key.rsplit("/", 1)[-1]
                assert values == pytest.approx(expected_by_input[filename]), (
                    f"{filename}: got {values!r}, expected {expected_by_input[filename]!r}"
                )
        finally:
            # Best-effort model cleanup; TransformJob is a completed record.
            try:
                Model.get(model_name=model_name, session=boto_session).delete()
            except Exception as e:
                LOGGER.warning(f"Cleanup Model {model_name} failed: {e}")
