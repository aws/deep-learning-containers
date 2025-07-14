import logging
import time
from invoke import run
from typing import Dict, List, Any
import boto3


from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class FsxSetup:
    """
    A utility class for setting up and managing FSx for Lustre filesystems
    and related AWS and Kubernetes resources.

    : param region: AWS region where resources will be created (default: "us-west-2")
    """

    def __init__(self, region: str = "us-west-2"):
        self.region = region

    def create_fsx_filesystem(
        self,
        subnet_id: str,
        security_group_ids: List[str],
        storage_capacity: int,
        deployment_type: str,
        tags: Dict[str, str],
    ):
        """
        Create FSx filesystem with given configuration
        : param subnet_id: subnet ID where FSx will be created
        : param security_group_ids: list of security group IDs
        : param storage_capacity: storage capacity in GiB
        : param deployment_type: FSx deployment type
        : param tags: dictionary of tags to apply to the FSx filesystem
        : return: dictionary containing filesystem details
        """
        tags_param = " ".join([f"Key={k},Value={v}" for k, v in tags.items()])

        try:
            fsx_id = run(
                f"aws fsx create-file-system"
                f" --file-system-type LUSTRE"
                f" --storage-capacity {storage_capacity}"
                f" --subnet-ids {subnet_id}"
                f' --security-group-ids {" ".join(security_group_ids)}'
                f" --lustre-configuration DeploymentType={deployment_type}"
                f" --tags {tags_param}"
                f' --query "FileSystem.FileSystemId"'
                f" --output text"
            ).stdout.strip()

            logger.info(f"Created FSx filesystem: {fsx_id}")

            filesystem_info = self.wait_for_filesystem(fsx_id)
            return filesystem_info

        except Exception as e:
            logger.error(f"Failed to create FSx filesystem: {e}")
            raise

    def delete_fsx_filesystem(self, fsx_id: str):

        try:
            fsx_id = run(
                f"aws fsx delete-file-system"
                f" --file-system-id {fsx_id}"
                f' --query "FileSystem.FileSystemId"'
                f" --output text"
            ).stdout.strip()

            print(f"Deleted FSx filesystem: {fsx_id}")

        except Exception as e:
            logger.error(f"Failed to create FSx filesystem: {e}")
            raise

    def wait_for_filesystem(self, filesystem_id: str):
        """
        Wait for FSx filesystem to become available and return its details
        : param filesystem_id: FSx filesystem ID
        : return: dictionary containing filesystem details (filesystem_id, dns_name, mount_name)
        : raises: Exception if filesystem enters FAILED, DELETING, or DELETED state
        """
        print(f"Waiting for FSx filesystem {filesystem_id} to be available...")
        while True:
            status = run(
                f"aws fsx describe-file-systems --file-system-id {filesystem_id} "
                f"--query 'FileSystems[0].Lifecycle' --output text"
            ).stdout.strip()

            if status == "AVAILABLE":
                break
            elif status in ["FAILED", "DELETING", "DELETED"]:
                raise Exception(f"FSx filesystem entered {status} state")

            print(f"FSx status: {status}, waiting...")
            time.sleep(30)

        # get fs DNS and mount name
        fsx_dns = run(
            f"aws fsx describe-file-systems --file-system-id {filesystem_id} "
            f"--query 'FileSystems[0].DNSName' --output text"
        ).stdout.strip()

        fsx_mount = run(
            f"aws fsx describe-file-systems --file-system-id {filesystem_id} "
            f"--query 'FileSystems[0].LustreConfiguration.MountName' --output text"
        ).stdout.strip()

        return {"filesystem_id": filesystem_id, "dns_name": fsx_dns, "mount_name": fsx_mount}

    def create_fsx_security_group(self, ec2_cli, vpc_id, group_name, description):
        """
        Create a security group for FSx Lustre and add inbound rules.

        :param vpc_id: The ID of the VPC where the security group will be created
        :param instance_id: The ID of the newly created EC2 instance
        :param region_name: The AWS region name
        :return: The ID of the created security group
        """
        try:
            # Create the security group
            response = ec2_cli.create_security_group(
                GroupName=group_name,
                Description=description,
                VpcId=vpc_id,
            )
            sg_id = response["GroupId"]
            print(f"Created security group: {sg_id}")

            return sg_id

        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    def add_ingress_rules_sg(self, ec2_cli, sg_id, instance_ids):
        """
        Add ingress rules to FSx security group for multiple instances

        Args:
            ec2_cli: boto3 EC2 client
            sg_id: ID of the FSx security group
            instance_ids: List of EC2 instance IDs
        """
        try:
            # Get security group IDs for all instances
            instance_sg_ids = set()
            for instance_id in instance_ids:
                response = ec2_cli.describe_instances(InstanceIds=[instance_id])
                sg_id_instance = response["Reservations"][0]["Instances"][0]["SecurityGroups"][0][
                    "GroupId"
                ]
                instance_sg_ids.add(sg_id_instance)

            instance_group_pairs = [{"GroupId": sg} for sg in instance_sg_ids]

            all_group_pairs = instance_group_pairs + [{"GroupId": sg_id}]

            # Add inbound rules
            ec2_cli.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 988,
                        "ToPort": 1023,
                        "UserIdGroupPairs": all_group_pairs,
                    }
                ],
            )
            print(
                f"Added inbound rules to FSx security group {sg_id} for instance security groups: {instance_sg_ids}"
            )

        except Exception as e:
            print(f"Error adding ingress rules: {str(e)}")
            raise

    def delete_security_group(self, group_id: str):
        """
        Create a security group in the specified VPC
        : param vpc_id: VPC ID where the security group will be created
        : param name: name of the security group
        : param description: description of the security group
        : return: created security group ID
        : raises: Exception if security group creation fails
        """
        try:
            sg_id = run(f"aws ec2 delete-security-group --group-id {group_id}").stdout.strip()
            print(f"Deleted security group: {sg_id}")

        except Exception as e:
            logger.error(f"Failed to create security group: {e}")
            raise

    def setup_csi_driver(self):
        """
        Install and configure the AWS FSx CSI Driver in the Kubernetes cluster
        : return: None
        : raises: Exception if driver installation or verification fails
        """
        try:
            logger.info("Installing AWS FSx CSI Driver...")
            run(
                "helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver/"
            )
            run("helm repo update")
            run(
                "helm install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver --namespace kube-system"
            )
            run(
                "kubectl wait --for=condition=ready pod -l app=fsx-csi-controller -n kube-system --timeout=300s"
            )

            self._verify_csi_driver()
            logger.info("FSx CSI Driver installed successfully")
        except Exception as e:
            logger.error(f"Failed to setup FSx CSI driver: {e}")
            raise

    def _verify_csi_driver(self):
        """
        Verify that FSx CSI driver pods are running correctly in the cluster
        : return: None
        : raises: Exception if driver pods are not found or not running
        """
        result = run("kubectl get pods -n kube-system | grep fsx")

        if "fsx-csi-controller" not in result.stdout or "fsx-csi-node" not in result.stdout:
            raise Exception("FSx CSI driver pods not found")

        fsx_pods = [
            line
            for line in result.stdout.split("\n")
            if ("fsx-csi-controller" in line or "fsx-csi-node" in line) and "Running" in line
        ]

        if not fsx_pods:
            raise Exception("No running FSx CSI driver pods found")

        logger.info(f"Found {len(fsx_pods)} running FSx CSI driver pods")

    def setup_kubernetes_resources(
        self, storage_class_file: str, pv_file: str, pvc_file: str, replacements: Dict[str, str]
    ):
        """
        Setup Kubernetes FSx resources using provided yaml files and replacements
        : param storage_class_file: path to the storage class yaml file
        : param pv_file: path to the persistent volume yaml file
        : param pvc_file: path to the persistent volume claim yaml file
        : param replacements: dictionary of placeholder replacements
                            Example: {"<subnet-id>": "subnet-xxx", "<sg-id>": "sg-xxx"}
        : return: None
        : raises: Exception if resource creation fails
        """
        try:
            for file_path in [storage_class_file, pv_file, pvc_file]:
                for key, value in replacements.items():
                    run(f"sed -i 's|{key}|{value}|g' {file_path}")

            for file_path in [storage_class_file, pv_file, pvc_file]:
                run(f"kubectl apply -f {file_path}")

            self.validate_kubernetes_resources()

        except Exception as e:
            logger.error(f"Failed to setup Kubernetes FSx resources: {e}")
            raise

    def validate_kubernetes_resources(self):
        """
        Validate that FSx Kubernetes resources are properly created and bound
        : return: True if all resources are validated successfully
        : raises: Exception if any resource validation fails
        """
        try:
            sc_result = run("kubectl get sc fsx-sc")
            if "fsx-sc" not in sc_result.stdout or "fsx.csi.aws.com" not in sc_result.stdout:
                raise Exception("FSx storage class not created correctly")

            pv_result = run("kubectl get pv fsx-lustre-pv")
            if "fsx-lustre-pv" not in pv_result.stdout or "Bound" not in pv_result.stdout:
                raise Exception("FSx persistent volume not created correctly")

            pvc_result = run("kubectl get pvc fsx-lustre-pvc")
            if "fsx-lustre-pvc" not in pvc_result.stdout or "Bound" not in pvc_result.stdout:
                raise Exception("FSx persistent volume claim not created correctly")

            logger.info("FSx Kubernetes resources validated successfully")
            return True

        except Exception as e:
            logger.error(f"FSx resource validation failed: {e}")
            raise
