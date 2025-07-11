import logging
import time
import boto3
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
        self.fsx_client = boto3.client('fsx', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)

    def create_fsx_filesystem(
        self,
        subnet_id: str,
        security_group_ids: List[str],
        storage_capacity: int,
        deployment_type: str,
        tags: Dict[str, str],
    ):
        """
        Create FSx Lustre filesystem with given configuration
        : param subnet_id: subnet ID where FSx will be created
        : param security_group_ids: list of security group IDs
        : param storage_capacity: storage capacity in GiB
        : param deployment_type: FSx deployment type
        : param tags: dictionary of tags to apply to the FSx filesystem
        : return: dictionary containing filesystem details
        """
        try:
            response = self.fsx_client.create_file_system(
                FileSystemType='LUSTRE',
                StorageCapacity=storage_capacity,
                SubnetIds=[subnet_id],
                SecurityGroupIds=security_group_ids,
                LustreConfiguration={'DeploymentType': deployment_type},
                Tags=[{'Key': k, 'Value': v} for k, v in tags.items()]
            )
            
            filesystem_id = response['FileSystem']['FileSystemId']
            logger.info(f"Created FSx filesystem: {filesystem_id}")
            
            return self.wait_for_filesystem(filesystem_id)
            
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
        logger.info(f"Waiting for FSx filesystem {filesystem_id} to be available...")
        
        try:
            waiter = self.fsx_client.get_waiter('file_system_available')
            waiter.wait(
                FileSystemIds=[filesystem_id],
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )

            # Get filesystem details
            response = self.fsx_client.describe_file_systems(
                FileSystemIds=[filesystem_id]
            )
            filesystem = response['FileSystems'][0]

            return {
                'filesystem_id': filesystem_id,
                'dns_name': filesystem['DNSName'],
                'mount_name': filesystem['LustreConfiguration']['MountName']
            }

        except Exception as e:
            logger.error(f"Error waiting for filesystem {filesystem_id}: {e}")
            raise

    def create_security_group(
        self,
        vpc_id: str,
        name: str,
        description: str
    ):
        """
        Create a security group in the specified VPC
        : param vpc_id: VPC ID where the security group will be created
        : param name: name of the security group
        : param description: description of the security group
        : return: created security group ID
        : raises: Exception if security group creation fails
        """
        try:
            response = self.ec2_client.create_security_group(
                GroupName=name,
                Description=description,
                VpcId=vpc_id
            )
            sg_id = response['GroupId']
            logger.info(f"Created security group: {sg_id}")
            return sg_id

        except Exception as e:
            logger.error(f"Failed to create security group: {e}")
            raise

    def add_security_group_ingress_rules(
        self,
        security_group_id: str,
        ingress_rules: List[Dict[str, Any]]
    ):
        """
        Add ingress rules to an existing security group
        : param security_group_id: ID of the security group to modify
        : param ingress_rules: list of dictionaries containing ingress rule configurations
                            Example: [{"protocol": "tcp", "port": "988-1023", "source-group": "sg-xxx"}]
        : return: None
        : raises: Exception if adding ingress rules fails
        """
        try:
            ip_permissions = []
            for rule in ingress_rules:
                from_port, to_port = map(int, rule['port'].split('-'))
                permission = {
                    'IpProtocol': rule['protocol'],
                    'FromPort': from_port,
                    'ToPort': to_port,
                    'UserIdGroupPairs': [{
                        'GroupId': rule['source-group']
                    }]
                }
                ip_permissions.append(permission)

            self.ec2_client.authorize_security_group_ingress(
                GroupId=security_group_id,
                IpPermissions=ip_permissions
            )
            
            logger.info(f"Added ingress rules to security group: {security_group_id}")

        except Exception as e:
            logger.error(f"Failed to add ingress rules to security group: {e}")
            raise

    def setup_csi_driver(self):
        """
        Install and configure the AWS FSx CSI Driver in the Kubernetes cluster
        : return: None           
        : raises: Exception if driver installation or verification fails
        """
        try:
            logger.info("Installing AWS FSx CSI Driver...")
            run("helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver/")
            run("helm repo update")
            run("helm install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver --namespace kube-system")
            run("kubectl wait --for=condition=ready pod -l app=fsx-csi-controller -n kube-system --timeout=300s")
            
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
            line for line in result.stdout.split("\n")
            if ("fsx-csi-controller" in line or "fsx-csi-node" in line) and "Running" in line
        ]
        
        if not fsx_pods:
            raise Exception("No running FSx CSI driver pods found")
        
        logger.info(f"Found {len(fsx_pods)} running FSx CSI driver pods")

    def setup_kubernetes_resources(
        self,
        storage_class_file: str,
        pv_file: str,
        pvc_file: str,
        replacements: Dict[str, str]
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