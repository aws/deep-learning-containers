"""SageMaker Batch Transform integration test for TF 2.20 inference DLC.

Runs a batch transform job against the DLC image: upload input JSON objects
to S3, create a Model + TransformJob, wait for completion, download the
output objects and assert the multiplier was applied.

Batch transform reuses the exact same handler stack as real-time endpoints
(nginx + gunicorn + python_service.py + tensorflow_model_server), so this
test's primary value is exercising the SageMaker-side ``CreateTransformJob``
wiring and the DLC's request-per-file behaviour — a code path the
real-time single-model / MME tests do not touch.

Covers audit finding "batch transform" — master TF 2.19 had
``test_batch_transform`` on the local test tier that this PR does not port.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


def test_batch_transform_json(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    inference_image_uri,
    sm_instance_type,
    unique_name,
):
    """End-to-end batch transform on JSON inputs. Uploads 3 single-record
    JSON files, runs the job, then downloads and verifies each output file
    contains predictions matching ``input * 2.0``."""
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
    run_id = unique_name("batch")

    with tempfile.TemporaryDirectory(prefix="tf220-batch-") as workdir:
        workdir_path = Path(workdir)

        # 1. Build + upload model.
        tar_path = build_sample_model(output_dir=workdir_path, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/batch/{run_id}/model",
        )

        # 2. Write 3 input files (one JSON body per file — the simplest
        #    batch split strategy: SplitType=None, one request per S3 object).
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
        model_name = unique_name("tf220-batch-model")
        job_name = unique_name("tf220-batch-job")
        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=inference_image_uri,
                model_data_url=model_data,
            ),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        try:
            # 4. Kick off the transform job.
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
                    instance_type=sm_instance_type,
                    instance_count=1,
                ),
                session=boto_session,
            )
            job.wait_for_status("Completed")

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
                values = (
                    first["output"]
                    if isinstance(first, dict) and "output" in first
                    else first
                )
                filename = key.rsplit("/", 1)[-1]
                assert values == pytest.approx(expected_by_input[filename]), (
                    f"{filename}: got {values!r}, expected {expected_by_input[filename]!r}"
                )
        finally:
            # Best-effort cleanup — TransformJob is a completed record, only
            # model needs deletion; output S3 objects survive by design.
            try:
                Model.get(model_name=model_name, session=boto_session).delete()
            except Exception:
                pass
