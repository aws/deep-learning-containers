#!/usr/bin/env python3

import os
import sys
import time
import logging
import boto3
import uuid
from invoke import run
from test_utils import eks as eks_utils
from test_utils import ec2 as ec2_utils
from test_utils import (
    generate_ssh_keypair,
    destroy_ssh_keypair,
    get_dlami_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EksInfrastructure:
    def __init__(self):
        self.cluster_name = "vllm-cluster"
        self.region = os.getenv("AWS_REGION", "us-west-2")
        self.instance_id = None
        self.key_filename = None
        self.ec2_client = None
        self.connection = None

    def setup_infrastructure(self):
        try:
            self.ec2_client = boto3.client("ec2", region_name=self.region)
            key_name = f"vllm-eks-test-{str(uuid.uuid4())}"
            self.key_filename = generate_ssh_keypair(self.ec2_client, key_name)
            
            # launch EC2 instance
            ami_id = get_dlami_id(self.region)
            instance_type = ec2_utils.get_ec2_instance_type("c5.12xlarge", "cpu")[0]
            instance_info = ec2_utils.launch_instance(
                ami_id=ami_id,
                instance_type=instance_type,
                ec2_key_name=key_name,
                region=self.region,
                iam_instance_profile_name=ec2_utils.EC2_INSTANCE_ROLE_NAME,
                instance_name="vLLM-EKS-Integration-Test",
            )
            self.instance_id = instance_info["InstanceId"]
            
            # setup connection
            ec2_utils.check_instance_state(self.instance_id, region=self.region)
            ec2_utils.check_system_state(self.instance_id, region=self.region)
            self.connection = ec2_utils.get_ec2_fabric_connection(
                self.instance_id, self.key_filename, self.region
            )
            
            # install prerequisites
            self.connection.run("pip3 install --user boto3 invoke packaging")
            self.validate_required_tools()
            self.setup_prerequisites()
            self.create_eks_cluster()
            self.validate_cluster_setup()
            self.setup_fsx_lustre()
            self.setup_load_balancer_controller()
            
            return True
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            self.cleanup_infrastructure()
            return False


    def setup_eks_tools(self):
            logger.info("Setting up EKS tools...")
            # use existing setup for eksctl, kubectl, aws-iam-authenticator
            eks_utils.eks_setup()
            # install helm separately
            self.install_helm()

            logger.info("EKS tools setup completed")

    def install_helm(self):
        logger.info("Installing Helm...")
        result = run("which helm", warn=True)
        if result.return_code == 0:
            logger.info("Helm already installed")
            return
        run(
            "curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3"
        )
        run("chmod 700 get_helm.sh")
        run("sudo ./get_helm.sh")
        run("rm -f get_helm.sh")

        result = run("which helm", warn=True)
        if result.return_code != 0:
            raise Exception("Helm installation failed - helm not found in PATH")

        logger.info("Helm installed successfully")
        
    def validate_required_tools(self):
        """
        Validate required tools and handle installation
        """
        logger.info("Validating required tools...")
        required_tools = ["aws", "eksctl", "kubectl", "helm", "curl", "jq"]
        missing_tools = []

        for tool in required_tools:
            result = run(f"which {tool}", warn=True)
            if result.return_code != 0:
                missing_tools.append((tool, tool.upper()))
                logger.warning(f"{tool} not found")
            else:
                logger.info(f"{tool} found: {result.stdout.strip()}")

        if missing_tools:
            logger.info("Installing missing tools...")
            self.setup_eks_tools()
            logger.info("Tools installed successfully")
        else:
            logger.info("All required tools are available")


    def validate_aws_credentials(self):
        """
        Validate AWS credentials and set required IAM roles for EKS
        """
        logger.info("Validating AWS credentials...")
        try:
            sts_client = boto3.client("sts")
            identity = sts_client.get_caller_identity()
            logger.info(f"AWS Identity validated: {identity['Arn']}")

            if not eks_utils.get_eks_role():
                os.environ["EKS_TEST_ROLE"] = identity["Arn"]
                logger.info(f"Set EKS_TEST_ROLE: {identity['Arn']}")

            return True
        except Exception as e:
            logger.error(f"AWS credential validation failed: {e}")
            return False


    def setup_prerequisites(self):
        """
        Setup required tools and repositories
        """
        logger.info("Setting up prerequisites...")

        run("pip install --quiet git-remote-codecommit")
        run("git config --global --add protocol.codecommit.allow always")
        run("git clone codecommit::us-west-2://aws-vllm-dlc-blog-repo aws-vllm-dlc-blog-repo")


    def create_eks_cluster(self):
        """
        Create EKS cluster and setup IAM access
        """
        logger.info("Creating EKS cluster...")

        run(
            f"cd aws-vllm-dlc-blog-repo && eksctl create cluster -f eks-cluster.yaml --region {self.region}"
        )

        # create a node group with EFA Support
        run(
            f"cd aws-vllm-dlc-blog-repo && eksctl create nodegroup -f large-model-nodegroup.yaml --region {self.region}"
        )

        eks_utils.eks_write_kubeconfig(self.cluster_name, self.region)

        # verify that nodes are ready
        result = run("kubectl get nodes")
        assert "Ready" in result.stdout, "EKS nodes not ready"
        logger.info("EKS cluster created successfully")


    def validate_cluster_setup(self):
        """
        Validate cluster setup including NVIDIA device plugin
        """
        logger.info("Validating cluster setup...")

        if not eks_utils.is_eks_cluster_active(self.cluster_name):
            raise Exception(f"EKS cluster {self.cluster_name} is not active")

        # check NVIDIA device plugin pods
        logger.info("Checking NVIDIA device plugin pods...")
        result = run("kubectl get pods -n kube-system | grep nvidia")

        if "nvidia-device-plugin" not in result.stdout:
            raise Exception("NVIDIA device plugin pods not found")

        # count running NVIDIA pods
        nvidia_pods = [
            line
            for line in result.stdout.split("\n")
            if "nvidia-device-plugin" in line and "Running" in line
        ]
        logger.info(f"Found {len(nvidia_pods)} running NVIDIA device plugin pods")

        if not nvidia_pods:
            raise Exception("No running NVIDIA device plugin pods found")

        # verify GPUs are available
        result = run("kubectl get nodes -o json | jq '.items[].status.capacity.\"nvidia.com/gpu\"'")
        gpu_counts = [
            line.strip().strip('"')
            for line in result.stdout.split("\n")
            if line.strip() and line.strip() != "null"
        ]

        if not gpu_counts:
            raise Exception("No GPUs found in cluster nodes")

        total_gpus = sum(int(count) for count in gpu_counts if count.isdigit())
        logger.info(f"Total GPUs available in cluster: {total_gpus}")

        if total_gpus == 0:
            raise Exception("No GPUs available in cluster")

        logger.info("Cluster setup validation completed")


    def setup_fsx_lustre(self):
        """
        Setup FSx Lustre filesystem with complete configuration
        """
        logger.info("Setting up FSx Lustre filesystem...")

        vpc_id = run(
            f"aws eks describe-cluster --name {self.cluster_name} --query 'cluster.resourcesVpcConfig.vpcId' --output text"
        ).stdout.strip()
        logger.info(f"Using VPC: {vpc_id}")

        subnet_id = run(
            f"aws eks describe-cluster --name {self.cluster_name} --query 'cluster.resourcesVpcConfig.subnetIds[0]' --output text"
        ).stdout.strip()
        logger.info(f"Using subnet: {subnet_id}")

        cluster_sg_id = run(
            f"aws eks describe-cluster --name {self.cluster_name} --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text"
        ).stdout.strip()
        logger.info(f"Using cluster security group: {cluster_sg_id}")

        # create security group for FSx Lustre
        sg_id = run(
            f'aws ec2 create-security-group --group-name fsx-lustre-sg --description "Security group for FSx Lustre" --vpc-id {vpc_id} --query "GroupId" --output text'
        ).stdout.strip()

        # add inbound rules for FSx Lustre
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 988-1023 --source-group {cluster_sg_id}"
        )
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 988-1023 --source-group {sg_id}"
        )

        # create FSx filesystem
        fsx_id = run(
            f'aws fsx create-file-system --file-system-type LUSTRE --storage-capacity 1200 --subnet-ids {subnet_id} --security-group-ids {sg_id} --lustre-configuration DeploymentType=SCRATCH_2 --tags Key=Name,Value=vllm-model-storage --query "FileSystem.FileSystemId" --output text'
        ).stdout.strip()

        logger.info("Waiting for FSx filesystem to be available...")
        while True:
            status = run(
                f"aws fsx describe-file-systems --file-system-id {fsx_id} --query 'FileSystems[0].Lifecycle' --output text"
            ).stdout.strip()
            if status == "AVAILABLE":
                break
            logger.info(f"FSx status: {status}, waiting...")
            time.sleep(30)

        # get FSx DNS and mount name
        fsx_dns = run(
            f"aws fsx describe-file-systems --file-system-id {fsx_id} --query 'FileSystems[0].DNSName' --output text"
        ).stdout.strip()

        fsx_mount = run(
            f"aws fsx describe-file-systems --file-system-id {fsx_id} --query 'FileSystems[0].LustreConfiguration.MountName' --output text"
        ).stdout.strip()

        logger.info(f"FSx DNS: {fsx_dns}")
        logger.info(f"FSx Mount Name: {fsx_mount}")

        # install AWS FSx CSI Driver
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

        # verify FSx CSI driver pods are running
        logger.info("Checking FSx CSI driver pods...")
        result = run("kubectl get pods -n kube-system | grep fsx")

        if "fsx-csi-controller" not in result.stdout or "fsx-csi-node" not in result.stdout:
            raise Exception("FSx CSI driver pods not found")

        # count running FSx pods
        fsx_pods = [
            line
            for line in result.stdout.split("\n")
            if ("fsx-csi-controller" in line or "fsx-csi-node" in line) and "Running" in line
        ]
        logger.info(f"Found {len(fsx_pods)} running FSx CSI driver pods")

        if not fsx_pods:
            raise Exception("No running FSx CSI driver pods found")

        logger.info("FSx CSI driver verification completed")

        # create Kubernetes resources for FSx Lustre
        run(
            f"cd aws-vllm-dlc-blog-repo && sed -i 's|<subnet-id>|{subnet_id}|g' fsx-storage-class.yaml"
        )
        run(f"cd aws-vllm-dlc-blog-repo && sed -i 's|<sg-id>|{sg_id}|g' fsx-storage-class.yaml")
        run(f"cd aws-vllm-dlc-blog-repo && sed -i 's|<fs-id>|{fsx_id}|g' fsx-lustre-pv.yaml")
        run(
            f"cd aws-vllm-dlc-blog-repo && sed -i 's|<fs-id>.fsx.us-west-2.amazonaws.com|{fsx_dns}|g' fsx-lustre-pv.yaml"
        )
        run(
            f"cd aws-vllm-dlc-blog-repo && sed -i 's|<mount-name>|{fsx_mount}|g' fsx-lustre-pv.yaml"
        )

        # apply FSx Kubernetes resources
        logger.info("Creating FSx Kubernetes storage resources...")
        run("cd aws-vllm-dlc-blog-repo && kubectl apply -f fsx-storage-class.yaml")
        run("cd aws-vllm-dlc-blog-repo && kubectl apply -f fsx-lustre-pv.yaml")
        run("cd aws-vllm-dlc-blog-repo && kubectl apply -f fsx-lustre-pvc.yaml")

        # make sure storage resources are created correctly
        logger.info("Validating FSx storage resources...")

        # check storage class
        sc_result = run("kubectl get sc fsx-sc")
        if "fsx-sc" not in sc_result.stdout or "fsx.csi.aws.com" not in sc_result.stdout:
            raise Exception("FSx storage class not created correctly")
        logger.info("FSx storage class created")

        # check persistent volume
        pv_result = run("kubectl get pv fsx-lustre-pv")
        if "fsx-lustre-pv" not in pv_result.stdout or "Bound" not in pv_result.stdout:
            raise Exception("FSx persistent volume not created correctly")
        logger.info("FSx persistent volume created and bound")

        # check persistent volume claim
        pvc_result = run("kubectl get pvc fsx-lustre-pvc")
        if "fsx-lustre-pvc" not in pvc_result.stdout or "Bound" not in pvc_result.stdout:
            raise Exception("FSx persistent volume claim not created correctly")
        logger.info("FSx persistent volume claim created and bound")

        logger.info("FSx Lustre setup and validation completed")


    def setup_load_balancer_controller(self):
        logger.info("Setting up AWS Load Balancer Controller...")
        run("helm repo add eks https://aws.github.io/eks-charts")
        run("helm repo update")
        run("kubectl apply -f https://raw.githubusercontent.com/aws/eks-charts/master/stable/aws-load-balancer-controller/crds/crds.yaml")
        run(
            f"helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName={self.cluster_name} --set serviceAccount.create=false --set enableServiceMutatorWebhook=false"
        )
        # install LeaderWorkerSet controller
        run(
            "helm install lws oci://registry.k8s.io/lws/charts/lws --version=0.6.1 --namespace lws-system --create-namespace --wait --timeout 300s"
        )
        # wait for controllers to be ready
        run(
            "kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=aws-load-balancer-controller -n kube-system --timeout=300s"
        )
        # setup sg for ALB
        user_ip = run("curl -s https://checkip.amazonaws.com").stdout.strip()
        vpc_id = run(
            f"aws eks describe-cluster --name {self.cluster_name} --query 'cluster.resourcesVpcConfig.vpcId' --output text"
        ).stdout.strip()
        # create ALB sg
        alb_sg = run(
            f'aws ec2 create-security-group --group-name vllm-alb-sg --description "Security group for vLLM ALB" --vpc-id {vpc_id} --query "GroupId" --output text'
        ).stdout.strip()
        # allow inbound traffic on port 80 from user IP
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {alb_sg} --protocol tcp --port 80 --cidr {user_ip}/32"
        )
        # get node sg
        node_instance_id = run(
            'aws ec2 describe-instances --filters "Name=tag:eks:nodegroup-name,Values=vllm-p4d-nodes-efa" --query "Reservations[0].Instances[0].InstanceId" --output text'
        ).stdout.strip()
        node_sg = run(
            f"aws ec2 describe-instances --instance-ids {node_instance_id} --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text"
        ).stdout.strip()
        # allow traffic from ALB to nodes on port 8000
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {node_sg} --protocol tcp --port 8000 --source-group {alb_sg}"
        )
        # update the sg in the ingress file
        run(
            f"cd aws-vllm-dlc-blog-repo && sed -i 's|<sg-id>|{alb_sg}|g' vllm-deepseek-32b-lws-ingress.yaml"
        )
        
        # verify sg were created and configured correctly
        logger.info("Verifying security group configurations...")
        
        # verify ALB sg
        alb_sg_result = run(
            f'aws ec2 describe-security-groups --group-ids {alb_sg} --query "SecurityGroups[0].IpPermissions"'
        )
        if "80" not in alb_sg_result.stdout:
            raise Exception("ALB security group not configured correctly - missing port 80 rule")
        logger.info("ALB security group configured correctly")
        
        # verify node sg rules
        node_sg_result = run(
            f'aws ec2 describe-security-groups --group-ids {node_sg} --query "SecurityGroups[0].IpPermissions"'
        )
        if "8000" not in node_sg_result.stdout:
            raise Exception("Node security group not configured correctly - missing port 8000 rule")
        
        logger.info("Node security group configured correctly")

        logger.info("Load Balancer Controller setup and verification completed")


    def cleanup_resources(self):
        logger.info("Running cleanup script...")
        try:
            script_path = "test/vllm_tests/test/vllm_eks_cleanup.sh"
            run(f"chmod +x {script_path}")
            run(
                f"cd aws-vllm-dlc-blog-repo && echo 'y' | ../{script_path}",
                check=False,
                timeout=3600,
            )
            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


    def cleanup_infrastructure(self):
        try:
            if self.connection:
                self.cleanup_resources()
            
            if self.instance_id:
                ec2_utils.terminate_instance(self.instance_id, region=self.region)
            
            if self.key_filename:
                destroy_ssh_keypair(self.ec2_client, self.key_filename)
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
