import logging
import time
from invoke import run
from typing import Dict, List, Any

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

    def create_security_group(self, vpc_id: str, name: str, description: str):
        """
        Create a security group in the specified VPC
        : param vpc_id: VPC ID where the security group will be created
        : param name: name of the security group
        : param description: description of the security group
        : return: created security group ID
        : raises: Exception if security group creation fails
        """
        try:
            sg_id = run(
                f"aws ec2 create-security-group"
                f" --group-name {name}"
                f' --description "{description}"'
                f" --vpc-id {vpc_id}"
                f' --query "GroupId"'
                f" --output text"
            ).stdout.strip()
            print(f"Created security group: {sg_id}")
            return sg_id

        except Exception as e:
            logger.error(f"Failed to create security group: {e}")
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

    def add_fsx_security_group_rules(
        self, security_group_id: str, ec2_client, client_security_group_ids: list = None
    ):
        """
        Add security group rules for FSx Lustre file system
        :param security_group_id: ID of the FSx security group
        :param ec2_client: boto3 EC2 client
        :param client_security_group_ids: List of client security group IDs
        """
        try:
            # Add inbound rules
            inbound_rules = [
                # Rules for FSx-to-FSx traffic
                {
                    "IpProtocol": "tcp",
                    "FromPort": 988,
                    "ToPort": 988,
                    "UserIdGroupPairs": [{"GroupId": security_group_id}],
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 1018,
                    "ToPort": 1023,
                    "UserIdGroupPairs": [{"GroupId": security_group_id}],
                },
            ]

            # Add rules for client traffic if client security groups are provided
            if client_security_group_ids:
                client_rules = [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 988,
                        "ToPort": 988,
                        "UserIdGroupPairs": [
                            {"GroupId": sg_id} for sg_id in client_security_group_ids
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 1018,
                        "ToPort": 1023,
                        "UserIdGroupPairs": [
                            {"GroupId": sg_id} for sg_id in client_security_group_ids
                        ],
                    },
                ]
                inbound_rules.extend(client_rules)

            # Apply inbound rules
            ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id, IpPermissions=inbound_rules
            )

            # Add outbound rules (same as inbound rules)
            ec2_client.authorize_security_group_egress(
                GroupId=security_group_id, IpPermissions=inbound_rules
            )

            print(f"Successfully added FSx security group rules to: {security_group_id}")

        except ec2_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                print(f"Error adding FSx security group rules: {str(e)}")
                raise
            else:
                print("Some rules already exist, continuing...")

    def add_client_security_group_rules(
        self,
        client_security_group_id: str,
        ec2_client,
        fsx_security_group_ids: list,
        other_client_security_group_ids: list = None,
    ):
        """
        Add security group rules for Lustre clients
        :param client_security_group_id: ID of the client security group
        :param ec2_client: boto3 EC2 client
        :param fsx_security_group_ids: List of FSx security group IDs
        :param other_client_security_group_ids: List of other client security group IDs
        """
        try:
            rules = []

            # Rules for client-to-client traffic
            if other_client_security_group_ids:
                rules.extend(
                    [
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 988,
                            "ToPort": 988,
                            "UserIdGroupPairs": [
                                {"GroupId": sg_id} for sg_id in other_client_security_group_ids
                            ],
                        },
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 1018,
                            "ToPort": 1023,
                            "UserIdGroupPairs": [
                                {"GroupId": sg_id} for sg_id in other_client_security_group_ids
                            ],
                        },
                    ]
                )

            # Rules for client-to-FSx traffic
            rules.extend(
                [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 988,
                        "ToPort": 988,
                        "UserIdGroupPairs": [
                            {"GroupId": sg_id} for sg_id in fsx_security_group_ids
                        ],
                    },
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 1018,
                        "ToPort": 1023,
                        "UserIdGroupPairs": [
                            {"GroupId": sg_id} for sg_id in fsx_security_group_ids
                        ],
                    },
                ]
            )

            # Apply inbound rules
            ec2_client.authorize_security_group_ingress(
                GroupId=client_security_group_id, IpPermissions=rules
            )

            # Apply outbound rules (same as inbound)
            ec2_client.authorize_security_group_egress(
                GroupId=client_security_group_id, IpPermissions=rules
            )

            print(f"Successfully added client security group rules to: {client_security_group_id}")

        except ec2_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                print(f"Error adding client security group rules: {str(e)}")
                raise
            else:
                print("Some rules already exist, continuing...")

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
