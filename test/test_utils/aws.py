# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""AWS Session Manager for all AWS boto3 API resources"""

import logging
import os
import stat
import tempfile
from datetime import datetime

import boto3
from fabric import Connection
from test_utils import random_suffix_name
from test_utils.constants import DEFAULT_REGION, EC2_INSTANCE_ROLE_NAME

LOGGER = logging.getLogger(__name__)


class LoggedConnection(Connection):
    """Fabric Connection that logs commands before execution."""

    def run(self, cmd, **kwargs):
        kwargs.setdefault("hide", True)
        kwargs.setdefault("in_stream", False)
        LOGGER.info(f"Running on {self.host}: {cmd}")
        return super().run(cmd, **kwargs)


class AWSSessionManager:
    def __init__(self, region=DEFAULT_REGION, profile_name=None):
        if profile_name:
            self.session = boto3.Session(profile_name=profile_name, region_name=region)
        else:
            self.session = boto3.Session(region_name=region)
        self.region = region

        # Client API
        self.cloudwatch = self.session.client("cloudwatch")
        self.codebuild = self.session.client("codebuild")
        self.codepipeline = self.session.client("codepipeline")
        self.ec2 = self.session.client("ec2")
        self.events = self.session.client("events")
        self.iam = self.session.client("iam")
        self.resource_groups = self.session.client("resource-groups")
        self.scheduler = self.session.client("scheduler")
        self.secretsmanager = self.session.client("secretsmanager")
        self.sts = self.session.client("sts")
        self.s3 = self.session.client("s3")
        self.sagemaker = self.session.client("sagemaker")
        self.ecr = self.session.client("ecr")

        # Resource API
        self.iam_resource = self.session.resource("iam")
        self.s3_resource = self.session.resource("s3")
        self.ssm = self.session.client("ssm")

    # ===========================================
    # ===== EC2 Instance Lifecycle ==============
    # ===========================================

    def get_latest_ami(
        self,
        parameter="/aws/service/deeplearning/ami/x86_64/base-with-single-cuda-amazon-linux-2023/latest/ami-id",
        before_date=None,
    ):
        """Resolve latest AMI ID via SSM parameter.

        Args:
            parameter: SSM parameter path for AMI lookup.
            before_date: If set (YYYY-MM-DD), use describe_images to find the latest AMI
                created before this date. Useful when the latest AMI is faulty.
        """
        if not before_date:
            response = self.ssm.get_parameter(Name=parameter)
            return response["Parameter"]["Value"]

        # Hotfix path: get latest AMI before a specific date
        ami_list = self.ec2.describe_images(
            Owners=["amazon"],
            Filters=[{"Name": "name", "Values": ["al2023-ami-2023.*-kernel-*-x86_64"]}],
        )
        filtered = [
            img
            for img in ami_list["Images"]
            if datetime.strptime(img["CreationDate"], "%Y-%m-%dT%H:%M:%S.%fZ")
            < datetime.strptime(before_date, "%Y-%m-%d")
        ]
        latest = max(filtered, key=lambda x: x["CreationDate"])
        LOGGER.info(
            f"Resolved AMI {latest['ImageId']} (created {latest['CreationDate']}, before {before_date})"
        )
        return latest["ImageId"]

    def launch_instance(
        self,
        ami_id,
        instance_type,
        key_name,
        instance_name="",
        iam_role=EC2_INSTANCE_ROLE_NAME,
        security_group_ids=None,
    ):
        """Launch a single EC2 instance with IMDSv2 enforced and 150GB EBS volume."""
        params = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "KeyName": key_name,
            "MinCount": 1,
            "MaxCount": 1,
            "MetadataOptions": {
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
                "HttpPutResponseHopLimit": 2,
            },
            "BlockDeviceMappings": [
                {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 150}},
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": f"CI-CD {instance_name}"}],
                },
            ],
        }
        if iam_role:
            params["IamInstanceProfile"] = {"Name": iam_role}
        if security_group_ids:
            params["SecurityGroupIds"] = security_group_ids

        response = self.ec2.run_instances(**params)
        instance_id = response["Instances"][0]["InstanceId"]
        LOGGER.info(f"Launched instance {instance_id} ({instance_type})")
        return instance_id

    def terminate_instance(self, instance_id):
        """Terminate an EC2 instance."""
        self.ec2.terminate_instances(InstanceIds=[instance_id])
        LOGGER.info(f"Terminated instance {instance_id}")

    def wait_for_instance_ready(self, instance_id, timeout=900):
        """Wait for instance to be running with status checks OK."""
        LOGGER.info(f"Waiting for instance {instance_id} to be ready...")
        waiter = self.ec2.get_waiter("instance_status_ok")
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 15, "MaxAttempts": timeout // 15},
        )
        LOGGER.info(f"Instance {instance_id} is ready")

    def get_public_ip(self, instance_id):
        """Get the public IP address of an instance."""
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        return response["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    def get_instance_tags(self, instance_id):
        """Get tags for an EC2 instance as a {key: value} dict."""
        response = self.ec2.describe_tags(
            Filters=[{"Name": "resource-id", "Values": [instance_id]}]
        )
        return {tag["Key"]: tag["Value"] for tag in response["Tags"]}

    # ===========================================
    # ===== Security Groups =====================
    # ===========================================

    def create_ssh_security_group(self, group_name=None):
        """Create a security group allowing SSH from anywhere. Returns group ID."""
        if not group_name:
            group_name = random_suffix_name("dlc-ssh", 36)
        vpc_id = self.ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])[
            "Vpcs"
        ][0]["VpcId"]
        response = self.ec2.create_security_group(
            GroupName=group_name,
            Description="Ephemeral SSH access for DLC tests",
            VpcId=vpc_id,
        )
        sg_id = response["GroupId"]
        self.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                },
            ],
        )
        LOGGER.info(f"Created security group {sg_id} ({group_name})")
        return sg_id

    def delete_security_group(self, sg_id):
        """Delete a security group."""
        self.ec2.delete_security_group(GroupId=sg_id)
        LOGGER.info(f"Deleted security group {sg_id}")

    # ===========================================
    # ===== SSH Key Pair Management =============
    # ===========================================

    def create_key_pair(self, key_name=None):
        """Create an EC2 key pair and save the PEM file. Returns (key_name, key_path)."""
        if not key_name:
            key_name = random_suffix_name("dlc-test", 36)
        response = self.ec2.create_key_pair(KeyName=key_name, KeyFormat="pem", KeyType="ed25519")
        key_path = os.path.join(tempfile.gettempdir(), f"{key_name}.pem")
        with open(key_path, "w") as f:
            f.write(response["KeyMaterial"])
        os.chmod(key_path, stat.S_IRUSR)
        LOGGER.info(f"Created key pair {key_name} at {key_path}")
        return key_name, key_path

    def delete_key_pair(self, key_name, key_path=None):
        """Delete an EC2 key pair and its local PEM file."""
        self.ec2.delete_key_pair(KeyName=key_name)
        if key_path and os.path.exists(key_path):
            os.remove(key_path)
        LOGGER.info(f"Deleted key pair {key_name}")

    # ===========================================
    # ===== SSH Connection ======================
    # ===========================================

    def get_ssh_connection(self, instance_id, key_path, user="ec2-user"):
        """Create a Fabric SSH connection to an EC2 instance."""
        ip = self.get_public_ip(instance_id)
        return LoggedConnection(
            host=ip,
            user=user,
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )
