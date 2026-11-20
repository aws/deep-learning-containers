"""Delete the SageMaker resources one CI job created, even if the job was killed.

A step timeout or a cancelled run kills pytest before its fixtures tear down,
which leaves GPU endpoints running. conftest.ci_tags() tags every Model,
EndpointConfig and Endpoint the suite creates with ``dlc-ci-job=<DLC_CI_JOB_ID>``;
this script, run by an always() step after the tests, deletes everything that
carries this job's tag. Resources of other jobs or PRs (other tag values, or no
tag) are never touched.

Usage: python cleanup_ci_resources.py --region us-west-2   (reads DLC_CI_JOB_ID)
"""

import argparse
import os
import sys

import boto3
from botocore.exceptions import ClientError

TAG_KEY = "dlc-ci-job"
NAME_PREFIX = "tf220-"


def _tagged(sm, arn, job_id):
    try:
        tags = sm.list_tags(ResourceArn=arn).get("Tags", [])
    except ClientError as e:
        # The listing covers every job's tf220- resources: another job may delete
        # one between the list and this call (ValidationException), or the calls
        # may be throttled. Neither is this job's resource to act on, so skip it.
        print(f"skipping {arn}: {e.response['Error']['Code']}")
        return False
    return any(t["Key"] == TAG_KEY and t["Value"] == job_id for t in tags)


def _matching(sm, list_call, items_key, arn_key, name_key, job_id):
    names = []
    for page in sm.get_paginator(list_call).paginate(NameContains=NAME_PREFIX):
        for item in page[items_key]:
            if _tagged(sm, item[arn_key], job_id):
                names.append(item[name_key])
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    job_id = os.environ.get("DLC_CI_JOB_ID", "").strip()
    if not job_id:
        print("DLC_CI_JOB_ID not set: nothing to clean up")
        return 0
    sm = boto3.client("sagemaker", region_name=args.region)
    failures = 0
    plan = [
        (
            "list_endpoints",
            "Endpoints",
            "EndpointArn",
            "EndpointName",
            sm.delete_endpoint,
            "EndpointName",
        ),
        (
            "list_endpoint_configs",
            "EndpointConfigs",
            "EndpointConfigArn",
            "EndpointConfigName",
            sm.delete_endpoint_config,
            "EndpointConfigName",
        ),
        ("list_models", "Models", "ModelArn", "ModelName", sm.delete_model, "ModelName"),
    ]
    for list_call, items_key, arn_key, name_key, delete, param in plan:
        for name in _matching(sm, list_call, items_key, arn_key, name_key, job_id):
            try:
                delete(**{param: name})
                print(f"deleted {name_key[:-4].lower()} {name}")
            except Exception as e:  # keep going: delete as much as possible
                failures += 1
                print(f"::warning::could not delete {name}: {e}")
    print(f"cleanup for {TAG_KEY}={job_id} done ({failures} failures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
