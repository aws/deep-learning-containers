import os
from invoke.context import Context
import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


report_group_name = os.getenv("CODEBUILD_BUILD_ID").split(":")[0]+"-test_reports"
report_group_arn = ":".join(os.getenv("CODEBUILD_BUILD_ARN").split("/")[0].split(":")[:-1])+":report-group/"+report_group_name
LOGGER.info(f"{report_group_arn}")
LOGGER.info(f"{report_group_name}")
config = {
  "exportConfigType": "S3",
  "s3Destination": {
    "bucket": "dlinfra-dlc-cicd-performance",
    "path": "test",
  }
}
LOGGER.info(f"{config}")
Context().run(f"aws codebuild update-report-group --arn {report_group_arn} --export-config {config}")
